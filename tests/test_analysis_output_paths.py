import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_offline_pi_representations.py"
SPEC = importlib.util.spec_from_file_location("analyze_offline_pi_representations", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load analyze_offline_pi_representations module at {MODULE_PATH}")
analyze_offline_pi_representations = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze_offline_pi_representations
SPEC.loader.exec_module(analyze_offline_pi_representations)


def test_offline_pi_representation_analysis_default_output_dir():
    output_dir = analyze_offline_pi_representations.resolve_output_dir(
        Path("runs/offline_pi/pointmaze_phase1_seed0/models/final_model.zip"),
        Path("data/datasets/pointmaze/phase1_pi/probe_seed1"),
        None,
    )

    assert output_dir == Path("runs/offline_pi/pointmaze_phase1_seed0/analysis/representations/probe_seed1")


def test_offline_pi_representation_analysis_output_dir_override():
    output_dir = analyze_offline_pi_representations.resolve_output_dir(
        Path("runs/offline_pi/pointmaze_phase1_seed0/models/final_model.zip"),
        Path("data/datasets/pointmaze/phase1_pi/probe_seed1"),
        Path("custom/analysis"),
    )

    assert output_dir == Path("custom/analysis")
