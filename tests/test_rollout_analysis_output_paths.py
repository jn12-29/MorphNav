import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tests" / "analyze_rollout_data.py"
SPEC = importlib.util.spec_from_file_location("analyze_rollout_data", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load analyze_rollout_data module at {MODULE_PATH}")
analyze_rollout_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze_rollout_data
SPEC.loader.exec_module(analyze_rollout_data)


def test_rollout_analysis_default_output_dir_is_next_to_data_dir(tmp_path: Path):
    data_dir = tmp_path / "rollouts" / "best" / "data"

    output_dir = analyze_rollout_data.resolve_output_dir(str(data_dir), None)

    assert output_dir == str(tmp_path / "rollouts" / "best" / "analysis")


def test_rollout_analysis_output_dir_override():
    output_dir = analyze_rollout_data.resolve_output_dir("runs/example/rollouts/best/data", "custom/analysis")

    assert output_dir == "custom/analysis"


def test_rollout_analysis_direct_call_resolves_empty_output_dir(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "rollouts" / "best" / "data"
    monkeypatch.setattr(analyze_rollout_data.DataRecorder, "load_all_episodes", lambda data_dir, prefix: [])

    analyze_rollout_data.analyze_rollout_data(str(data_dir))

    assert (tmp_path / "rollouts" / "best" / "analysis").is_dir()
