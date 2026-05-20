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
