import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch as th

from components.offline_pi_runtime import (
    TensorBoardRunWriter,
    append_jsonl,
    dataset_metadata_summary,
    make_event,
    make_run_dirs,
    write_json_atomic,
)
from components.offline_pi_workflow import (
    _event_epoch_metrics,
    _first_step_loss_weight,
    _max_grad_norm,
    _offline_pi_optimizer_kwargs,
    _optimizer_name,
    run_offline_pi_workflow,
)
from components.pi_algo import PathIntegrationRecurrentPPO
from scripts.offline_pi_rehearsal import PI_ZOO_CONFIG_PATH, _validate_args, build_parser


def test_run_dirs_jsonl_and_atomic_json_contract(tmp_path: Path):
    dirs = make_run_dirs(tmp_path / "run")
    assert (tmp_path / "run" / "metrics").is_dir()
    assert (tmp_path / "run" / "models").is_dir()
    assert (tmp_path / "run" / "eval").is_dir()

    payload = make_event("run_start", phase="train", epoch=0, update=0, global_step=0)
    append_jsonl(dirs["metrics"] / "metrics.jsonl", payload)
    rows = (dirs["metrics"] / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["event"] == "run_start"
    assert row["phase"] == "train"
    assert row["timestamp"]

    write_json_atomic(dirs["root"] / "config.json", {"run_name": "test"})
    assert json.loads((dirs["root"] / "config.json").read_text(encoding="utf-8")) == {"run_name": "test"}


def test_dataset_metadata_summary_keeps_manifest_compact(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "dataset_metadata.json").write_text(
        json.dumps({"dataset_name": "sample", "num_episodes": 2}),
        encoding="utf-8",
    )
    (dataset_root / "manifest.json").write_text(
        json.dumps({"dataset_seed": 0, "episodes": [{"episode_id": 0}, {"episode_id": 1}]}),
        encoding="utf-8",
    )

    summary = dataset_metadata_summary(dataset_root)

    assert summary is not None
    assert summary["metadata"]["dataset_name"] == "sample"
    assert summary["manifest"]["dataset_seed"] == 0
    assert summary["manifest"]["episode_count"] == 2
    assert "episodes" not in summary["manifest"]


def test_tensorboard_writer_disabled_without_dependency(tmp_path: Path):
    writer = TensorBoardRunWriter(tmp_path / "tb", enabled=False)
    writer.add_scalar("train/loss_step", 1.0, 1)
    writer.add_text("run/config", "{}")
    writer.close()
    assert not writer.enabled
    assert writer.disabled_reason == "disabled by CLI"


def test_tensorboard_writer_falls_back_when_dependency_missing(tmp_path: Path):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("missing tensorboard")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        writer = TensorBoardRunWriter(tmp_path / "tb", enabled=True)

    assert not writer.enabled
    assert "TensorBoard dependencies are unavailable" in str(writer.disabled_reason)
    writer.add_scalar("train/loss_step", 1.0, 1)
    writer.add_text("run/config", "{}")
    writer.close()


def test_workflow_config_records_tensorboard_dependency_fallback(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    args = SimpleNamespace(
        mode="probe",
        dataset_root=dataset_root,
        probe_dataset_root=None,
        model_path=None,
        output_dir=None,
        run_name=None,
        learning_rate=1e-4,
        first_step_loss_weight=10.0,
        batch_size_sequences=2,
        max_seq_len=None,
        max_updates=None,
        epochs=1,
        seed=0,
        device="cpu",
        log_every_updates=0,
        eval_every_epochs=0,
        eval_at_start=False,
        eval_artifact_every_epochs=0,
        eval_gridscore_every_epochs=0,
        gridscore_n_bins=32,
        gridscore_max_steps=None,
        gridscore_top_k=8,
        checkpoint_every_epochs=0,
        save_final_checkpoint=False,
        tensorboard=True,
        tensorboard_log_dir=None,
    )
    probe_metrics = {
        "offline_pi/probe/loss": 1.0,
        "offline_pi/probe/localization_mse": 2.0,
        "offline_pi/probe/localization_rmse": 3.0,
        "offline_pi/probe/localization_mae": 4.0,
        "offline_pi/probe/x_mae": 5.0,
        "offline_pi/probe/y_mae": 6.0,
        "offline_pi/probe/first_localization_mse": 0.5,
        "offline_pi/probe/first_localization_rmse": 0.707,
        "offline_pi/probe/first_localization_mae": 0.6,
        "offline_pi/probe/first_x_mae": 0.7,
        "offline_pi/probe/first_y_mae": 0.8,
        "offline_pi/probe/first_localization_mse_ratio": 0.25,
        "offline_pi/probe/first_localization_mae_ratio": 0.15,
        "offline_pi/probe/steps": 7.0,
        "offline_pi/probe/first_step_count": 2.0,
    }
    real_import = __import__

    def fake_import(name, *import_args, **import_kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("missing tensorboard")
        return real_import(name, *import_args, **import_kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch("components.offline_pi_workflow.run_offline_pi_probe", return_value=probe_metrics),
    ):
        run_offline_pi_workflow(
            object(),
            args,
            output_dir=tmp_path / "run",
            run_name="run",
            fresh_model_settings=None,
        )

    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["tensorboard_requested"] is True
    assert config["tensorboard_enabled"] is False
    assert "TensorBoard dependencies are unavailable" in config["tensorboard_disabled_reason"]
    assert config["fresh_model_config_path"] is None
    assert config["first_step_loss_weight"] == 10.0
    assert config["optimizer"] == "adam"
    assert config["weight_decay"] == 0.0
    assert config["momentum"] == 0.0
    assert config["optimizer_kwargs"] == {"weight_decay": 0.0}
    assert config["max_grad_norm"] == 0.5


def test_epoch_event_samples_seen_is_cumulative():
    metrics = {
        "offline_pi/loss": 1.0,
        "offline_pi/loss_std": 0.1,
        "offline_pi/localization_mse": 2.0,
        "offline_pi/localization_mse_std": 0.2,
        "offline_pi/first_localization_mse": 0.5,
        "offline_pi/first_localization_mse_std": 0.05,
        "offline_pi/first_localization_mse_ratio": 0.25,
        "offline_pi/steps": 4.0,
        "offline_pi/first_step_count": 2.0,
        "offline_pi/sequence_count": 2.0,
        "offline_pi/updates": 1.0,
        "offline_pi/epoch_seconds": 0.5,
        "offline_pi/samples_seen": 4.0,
    }

    event_metrics = _event_epoch_metrics(metrics, samples_seen=12)

    assert event_metrics["offline_pi/steps"] == 4.0
    assert event_metrics["offline_pi/first_localization_mse_mean"] == 0.5
    assert event_metrics["offline_pi/first_localization_mse_ratio"] == 0.25
    assert event_metrics["offline_pi/first_step_count"] == 2.0
    assert event_metrics["offline_pi/samples_seen"] == 12.0


def test_offline_pi_cli_tensorboard_defaults_on():
    parser = build_parser()

    args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    disabled = parser.parse_args(
        ["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0", "--no-tensorboard"]
    )

    assert args.tensorboard is True
    assert disabled.tensorboard is False


def test_offline_pi_cli_first_step_loss_weight_default():
    parser = build_parser()

    args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])

    assert args.first_step_loss_weight == 10.0


def test_offline_pi_cli_max_grad_norm_default_zero_and_validation():
    parser = build_parser()

    default_args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    disabled_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--max-grad-norm",
            "0.0",
        ]
    )
    negative_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--max-grad-norm",
            "-0.1",
        ]
    )

    assert default_args.max_grad_norm == 0.5
    assert disabled_args.max_grad_norm == 0.0
    with pytest.raises(SystemExit) as exc_info:
        _validate_args(parser, negative_args)
    assert exc_info.value.code == 2


def test_offline_pi_cli_optimizer_defaults_and_choices():
    parser = build_parser()

    default_args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    adamw_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0.01",
        ]
    )
    sgd_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--optimizer",
            "sgd",
            "--weight-decay",
            "0.02",
            "--momentum",
            "0.9",
        ]
    )
    rmsprop_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--optimizer",
            "rmsprop",
            "--weight-decay",
            "0.03",
            "--momentum",
            "0.5",
        ]
    )

    assert default_args.optimizer == "adam"
    assert default_args.weight_decay == 0.0
    assert default_args.momentum == 0.0
    assert adamw_args.optimizer == "adamw"
    assert adamw_args.weight_decay == 0.01
    assert _optimizer_name(sgd_args) == "sgd"
    assert _offline_pi_optimizer_kwargs(sgd_args) == {"weight_decay": 0.02, "momentum": 0.9}
    assert _optimizer_name(rmsprop_args) == "rmsprop"
    assert _offline_pi_optimizer_kwargs(rmsprop_args) == {"weight_decay": 0.03, "momentum": 0.5}


def test_offline_pi_workflow_first_step_loss_weight_fallback():
    assert _first_step_loss_weight(SimpleNamespace()) == 10.0


def test_offline_pi_workflow_max_grad_norm_fallback():
    assert _max_grad_norm(SimpleNamespace()) == 0.5


def test_online_pi_algo_first_step_loss_weight_default():
    assert inspect.signature(PathIntegrationRecurrentPPO).parameters["pi_first_step_loss_weight"].default == 10.0


def test_offline_pi_cli_gridscore_defaults_off():
    parser = build_parser()

    args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])

    assert args.eval_gridscore_every_epochs == 0
    assert args.gridscore_n_bins == 32
    assert args.gridscore_max_steps is None
    assert args.gridscore_top_k == 8
    assert args.gridscore_positive_activations is False

    enabled_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--gridscore-positive-activations",
        ]
    )
    assert enabled_args.gridscore_positive_activations is True


def test_offline_pi_cli_config_path_defaults_and_override():
    parser = build_parser()

    default_args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    custom_args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--config-path",
            "custom/maze_pi.yml",
        ]
    )

    assert default_args.config_path == PI_ZOO_CONFIG_PATH
    assert custom_args.config_path == Path("custom/maze_pi.yml")


def test_workflow_probe_records_gridscore_metrics_and_summary(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    args = SimpleNamespace(
        mode="probe",
        dataset_root=dataset_root,
        probe_dataset_root=None,
        model_path=None,
        output_dir=None,
        run_name=None,
        learning_rate=1e-4,
        first_step_loss_weight=10.0,
        batch_size_sequences=2,
        max_seq_len=None,
        max_updates=None,
        epochs=1,
        seed=0,
        device="cpu",
        log_every_updates=0,
        eval_every_epochs=0,
        eval_at_start=False,
        eval_artifact_every_epochs=0,
        eval_gridscore_every_epochs=1,
        gridscore_n_bins=8,
        gridscore_max_steps=16,
        gridscore_top_k=2,
        gridscore_positive_activations=True,
        checkpoint_every_epochs=0,
        save_final_checkpoint=False,
        tensorboard=False,
        tensorboard_log_dir=None,
    )
    probe_metrics = {
        "offline_pi/probe/loss": 1.0,
        "offline_pi/probe/localization_mse": 2.0,
        "offline_pi/probe/localization_rmse": 3.0,
        "offline_pi/probe/localization_mae": 4.0,
        "offline_pi/probe/x_mae": 5.0,
        "offline_pi/probe/y_mae": 6.0,
        "offline_pi/probe/first_localization_mse": 0.5,
        "offline_pi/probe/first_localization_rmse": 0.707,
        "offline_pi/probe/first_localization_mae": 0.6,
        "offline_pi/probe/first_x_mae": 0.7,
        "offline_pi/probe/first_y_mae": 0.8,
        "offline_pi/probe/first_localization_mse_ratio": 0.25,
        "offline_pi/probe/first_localization_mae_ratio": 0.15,
        "offline_pi/probe/steps": 7.0,
        "offline_pi/probe/first_step_count": 2.0,
        "offline_pi/probe/sequence_count": 8.0,
    }
    gridscore_summary = {
        "best_grid_score": 0.25,
        "best_unit": 3,
        "mean_grid_score": 0.1,
        "valid_units": 4,
        "unit_count": 4,
        "num_steps": 16,
        "bounds": [0.0, 1.0, 0.0, 1.0],
        "n_bins": 8,
        "max_steps": 16,
        "max_units": None,
        "top_k": 2,
    }

    with (
        patch("components.offline_pi_workflow.run_offline_pi_probe", return_value=probe_metrics),
        patch("components.offline_pi_gridscore.export_gridscore_artifacts", return_value=gridscore_summary) as gridscore_mock,
    ):
        run_offline_pi_workflow(
            object(),
            args,
            output_dir=tmp_path / "run",
            run_name="run",
            fresh_model_settings=None,
        )

    gridscore_mock.assert_called_once()
    assert gridscore_mock.call_args.kwargs["max_units"] is None
    assert gridscore_mock.call_args.kwargs["gridscore_positive_activations"] is True
    probe_payload = json.loads((tmp_path / "run" / "metrics" / "probe_epoch_0000.json").read_text(encoding="utf-8"))
    assert probe_payload["offline_pi/probe/first_localization_mse"] == 0.5
    assert probe_payload["offline_pi/probe/gridscore/best"] == 0.25
    assert probe_payload["offline_pi/probe/gridscore/best_unit"] == 3.0
    assert probe_payload["offline_pi/probe/gridscore/valid_units"] == 4.0

    rows = [
        json.loads(row)
        for row in (tmp_path / "run" / "metrics" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    probe_event = next(row for row in rows if row["event"] == "probe")
    assert probe_event["offline_pi/probe/gridscore/mean"] == 0.1
    assert probe_event["gridscore_summary"] == gridscore_summary
    final_metrics = json.loads((tmp_path / "run" / "metrics" / "offline_pi_metrics.json").read_text(encoding="utf-8"))
    assert final_metrics["offline_pi/probe/gridscore/best"] == 0.25


def test_workflow_train_passes_offline_optimizer_settings(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    args = SimpleNamespace(
        mode="train",
        dataset_root=dataset_root,
        probe_dataset_root=None,
        model_path=None,
        output_dir=None,
        run_name=None,
        learning_rate=1e-4,
        optimizer="adamw",
        weight_decay=0.02,
        momentum=0.0,
        max_grad_norm=0.25,
        first_step_loss_weight=10.0,
        batch_size_sequences=2,
        max_seq_len=None,
        max_updates=1,
        epochs=1,
        seed=0,
        device="cpu",
        log_every_updates=1,
        eval_every_epochs=0,
        eval_at_start=False,
        eval_artifact_every_epochs=0,
        eval_gridscore_every_epochs=0,
        gridscore_n_bins=32,
        gridscore_max_steps=None,
        gridscore_top_k=8,
        checkpoint_every_epochs=0,
        save_final_checkpoint=False,
        tensorboard=False,
        tensorboard_log_dir=None,
    )
    train_metrics = {
        "offline_pi/loss": 1.0,
        "offline_pi/localization_mse": 2.0,
        "offline_pi/first_localization_mse": 0.5,
        "offline_pi/first_localization_mse_ratio": 0.25,
        "offline_pi/updates": 1.0,
        "offline_pi/final_epoch": 1.0,
        "offline_pi/steps": 4.0,
    }

    with patch("components.offline_pi_workflow.run_offline_pi_rehearsal", return_value=train_metrics) as train_mock:
        run_offline_pi_workflow(
            object(),
            args,
            output_dir=tmp_path / "run",
            run_name="run",
            fresh_model_settings=None,
        )

    train_mock.assert_called_once()
    assert train_mock.call_args.kwargs["optimizer_cls"] is th.optim.AdamW
    assert train_mock.call_args.kwargs["optimizer_kwargs"] == {"weight_decay": 0.02}
    assert train_mock.call_args.kwargs["max_grad_norm"] == 0.25
    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["optimizer"] == "adamw"
    assert config["weight_decay"] == 0.02
    assert config["momentum"] == 0.0
    assert config["optimizer_kwargs"] == {"weight_decay": 0.02}
    assert config["max_grad_norm"] == 0.25
