import importlib.util
from pathlib import Path
import sys

import numpy as np

from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
)
from components.dataset_gen.pointmaze_zarr_writer import write_shard

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pointmaze_dataset.py"
SPEC = importlib.util.spec_from_file_location("analyze_pointmaze_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load analyze_pointmaze_dataset module at {MODULE_PATH}")
analyze_pointmaze_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze_pointmaze_dataset
SPEC.loader.exec_module(analyze_pointmaze_dataset)


def _episode(length: int, offset: float = 0.0) -> dict:
    xy = np.stack(
        [
            np.linspace(-1.0 + offset, 1.0 + offset, length),
            np.linspace(-0.5, 0.5, length),
        ],
        axis=1,
    ).astype(np.float32)
    action = np.stack([np.sin(np.arange(length)), np.cos(np.arange(length))], axis=1).astype(np.float32)
    qvel = np.diff(xy, axis=0, prepend=xy[:1]).astype(np.float32)
    return {
        "obs": {
            "observation": qvel,
            "start_pos": np.repeat(xy[:1], length, axis=0),
            "achieved_goal": xy,
            "desired_goal": np.repeat(np.array([[1.0, 1.0]], dtype=np.float32), length, axis=0),
        },
        "action": action,
        "reward": np.zeros(length, dtype=np.float32),
        "terminated": np.array([False] * (length - 1) + [True], dtype=bool),
        "truncated": np.zeros(length, dtype=bool),
        "qpos": xy,
        "qvel": qvel,
        "goal": np.repeat(np.array([[1.0, 1.0]], dtype=np.float32), length, axis=0),
        "annotation": {
            "agent_xy": xy,
            "heading": np.zeros(length, dtype=np.float32),
            "goal_xy": np.repeat(np.array([[1.0, 1.0]], dtype=np.float32), length, axis=0),
            "relative_goal": np.repeat(np.array([[1.0, 1.0]], dtype=np.float32), length, axis=0) - xy,
        },
        "summary": {
            "episode_length": length,
            "return": 0.0,
            "terminated_reason": "terminated",
            "path_length": 1.0,
            "net_displacement": 1.0,
            "coverage_score": 1.0,
            "stuck_ratio": 0.0,
            "collision_ratio": 0.0,
            "goal_reached": True,
            "mean_speed": 0.1,
            "mean_turn_rate": 0.0,
        },
    }


def test_analyze_pointmaze_dataset_summarizes_and_plots(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_meta = {
        "dataset_schema": POINTMAZE_MUJOCO_ZARR_SCHEMA,
        "dataset_schema_version": POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
        "dataset_name": "pointmaze-test",
        "dataset_seed": 9,
        "policy_type": "GridCellRandomWalkForceDriver",
        "policy_params": {"speed_mean": 0.25},
        "env_kwargs": {"maze_map_name": "OPEN"},
    }
    write_shard(dataset_root, 0, [_episode(4), _episode(5, offset=0.1)], dataset_meta, storage_format="npz")

    data = analyze_pointmaze_dataset.load_dataset_distribution(dataset_root, max_preview_episodes=1)
    summary = analyze_pointmaze_dataset.summarize_distribution(
        data,
        n_bins=8,
        bounds=(-2.5, 2.5, -2.5, 2.5),
    )
    analyze_pointmaze_dataset.plot_distribution(
        data,
        tmp_path / "analysis",
        n_bins=8,
        bounds=(-2.5, 2.5, -2.5, 2.5),
    )
    assert analyze_pointmaze_dataset.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(tmp_path / "analysis_cli"),
            "--n-bins",
            "8",
        ]
    ) == 0

    assert summary["policy_type"] == "GridCellRandomWalkForceDriver"
    assert summary["num_episodes"] == 2
    assert summary["num_steps"] == 9
    assert summary["occupancy"]["nonempty_bins"] > 0
    assert (tmp_path / "analysis" / "occupancy.png").exists()
    assert (tmp_path / "analysis" / "action_hist.png").exists()
    assert (tmp_path / "analysis" / "trajectory_preview.png").exists()
    assert (tmp_path / "analysis_cli" / "dataset_distribution.json").exists()
    assert (tmp_path / "analysis_cli" / "occupancy.png").exists()
