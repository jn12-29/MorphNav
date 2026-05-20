import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from components.offline_pi_runtime import (
    TensorBoardRunWriter,
    append_jsonl,
    dataset_metadata_summary,
    make_event,
    make_run_dirs,
    write_json_atomic,
)
from components.offline_pi_workflow import _event_epoch_metrics, run_offline_pi_workflow
from scripts.offline_pi_rehearsal import build_parser


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
        "offline_pi/probe/steps": 7.0,
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
            fresh_model_settings={},
        )

    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["tensorboard_requested"] is True
    assert config["tensorboard_enabled"] is False
    assert "TensorBoard dependencies are unavailable" in config["tensorboard_disabled_reason"]


def test_epoch_event_samples_seen_is_cumulative():
    metrics = {
        "offline_pi/loss": 1.0,
        "offline_pi/loss_std": 0.1,
        "offline_pi/localization_mse": 2.0,
        "offline_pi/localization_mse_std": 0.2,
        "offline_pi/steps": 4.0,
        "offline_pi/sequence_count": 2.0,
        "offline_pi/updates": 1.0,
        "offline_pi/epoch_seconds": 0.5,
        "offline_pi/samples_seen": 4.0,
    }

    event_metrics = _event_epoch_metrics(metrics, samples_seen=12)

    assert event_metrics["offline_pi/steps"] == 4.0
    assert event_metrics["offline_pi/samples_seen"] == 12.0


def test_offline_pi_cli_tensorboard_defaults_on():
    parser = build_parser()

    args = parser.parse_args(["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    disabled = parser.parse_args(
        ["--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0", "--no-tensorboard"]
    )

    assert args.tensorboard is True
    assert disabled.tensorboard is False
