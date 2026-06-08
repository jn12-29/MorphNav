import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "offline_pi_rehearsal.py"
SPEC = importlib.util.spec_from_file_location("offline_pi_rehearsal_script", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load offline_pi_rehearsal script at {MODULE_PATH}")
offline_pi_rehearsal_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = offline_pi_rehearsal_script
SPEC.loader.exec_module(offline_pi_rehearsal_script)


def test_offline_pi_default_output_dir_uses_timestamped_run_name():
    with patch.object(offline_pi_rehearsal_script, "datetime") as datetime_mock:
        datetime_mock.now.return_value.strftime.return_value = "20260520_153045"
        output_dir = offline_pi_rehearsal_script.resolve_output_dir(None, None, seed=7)

    assert output_dir == Path("runs/offline_pi/pointmaze_phase1_seed7_20260520_153045")


def test_offline_pi_run_name_output_dir():
    output_dir = offline_pi_rehearsal_script.resolve_output_dir(None, "manual_run", seed=7)

    assert output_dir == Path("runs/offline_pi/manual_run")


def test_offline_pi_explicit_output_dir_override():
    output_dir = offline_pi_rehearsal_script.resolve_output_dir(Path("custom/run"), "manual_run", seed=7)

    assert output_dir == Path("custom/run")


def test_fresh_model_kwargs_loads_policy_settings_from_zoo_config():
    model_config = offline_pi_rehearsal_script.fresh_model_kwargs(
        learning_rate=3e-4,
        seed=11,
        device="cpu",
    )

    assert model_config["policy"] == "PathIntegrationMultiInputLstmPolicy"
    kwargs = model_config["kwargs"]
    assert kwargs["learning_rate"] == 3e-4
    assert kwargs["seed"] == 11
    assert kwargs["device"] == "cpu"
    assert kwargs["pi_first_step_loss_weight"] == 10.0
    assert kwargs["pi_target_key"] == "achieved_goal"
    assert kwargs["policy_kwargs"]["features_extractor_kwargs"] == {"drop_keys": ["achieved_goal"]}
    assert kwargs["policy_kwargs"]["lstm_hidden_size"] == 256
    assert kwargs["policy_kwargs"]["pi_init_state_key"] == "start_pos"
    assert "n_envs" not in kwargs
    assert "n_timesteps" not in kwargs


def test_fresh_model_kwargs_supports_torch_optimizer_in_zoo_config(tmp_path: Path):
    config_path = tmp_path / "maze_pi.yml"
    config_path.write_text(
        """
PointMaze:
  n_envs: 1
  n_timesteps: 100
  policy: "PathIntegrationMultiInputLstmPolicy"
  pi_target_key: "achieved_goal"
  policy_kwargs: "dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(drop_keys=['achieved_goal']),
    optimizer_class=th.optim.AdamW,
    optimizer_kwargs=dict(weight_decay=0.01))"
""",
        encoding="utf-8",
    )

    model_config = offline_pi_rehearsal_script.fresh_model_kwargs(
        learning_rate=1e-4,
        seed=0,
        device="cpu",
        config_path=config_path,
    )

    policy_kwargs = model_config["kwargs"]["policy_kwargs"]
    assert policy_kwargs["optimizer_class"] is th.optim.AdamW
    assert policy_kwargs["optimizer_kwargs"] == {"weight_decay": 0.01}


def test_fresh_model_settings_uses_cli_config_path():
    parser = offline_pi_rehearsal_script.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/rehearsal_seed0",
            "--config-path",
            "custom/maze_pi.yml",
        ]
    )

    with patch.object(offline_pi_rehearsal_script, "fresh_model_kwargs", return_value={"policy": "custom"}) as mocked:
        settings = offline_pi_rehearsal_script._fresh_model_settings(args)

    assert settings == {"policy": "custom"}
    mocked.assert_called_once_with(
        learning_rate=1e-4,
        seed=0,
        device="auto",
        config_path=Path("custom/maze_pi.yml"),
    )


def test_fresh_model_settings_skips_loaded_model_config():
    parser = offline_pi_rehearsal_script.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "probe",
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/probe_seed1",
            "--model-path",
            "runs/offline_pi/example/models/final_model.zip",
            "--config-path",
            "missing/maze_pi.yml",
        ]
    )

    with patch.object(offline_pi_rehearsal_script, "fresh_model_kwargs") as mocked:
        settings = offline_pi_rehearsal_script._fresh_model_settings(args)

    assert settings is None
    mocked.assert_not_called()
