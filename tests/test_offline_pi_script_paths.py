import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


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
    assert kwargs["pi_target_key"] == "achieved_goal"
    assert kwargs["policy_kwargs"]["features_extractor_kwargs"] == {"drop_keys": ["achieved_goal"]}
    assert kwargs["policy_kwargs"]["lstm_hidden_size"] == 256
    assert kwargs["policy_kwargs"]["pi_init_state_key"] == "start_pos"
    assert "n_envs" not in kwargs
    assert "n_timesteps" not in kwargs
