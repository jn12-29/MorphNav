import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

MODULE_PATH = Path(__file__).resolve().parents[1] / "components" / "dataset_gen" / "pointmaze_zarr_writer.py"
SPEC = importlib.util.spec_from_file_location("pointmaze_zarr_writer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load pointmaze_zarr_writer module at {MODULE_PATH}")
pointmaze_zarr_writer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pointmaze_zarr_writer
SPEC.loader.exec_module(pointmaze_zarr_writer)

write_dataset_metadata = pointmaze_zarr_writer.write_dataset_metadata
write_shard = pointmaze_zarr_writer.write_shard


def _synthetic_episodes() -> list[dict]:
    episodes = [
        {
            "action": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "reward": np.array([1.0, 2.0], dtype=np.float32),
            "terminated": np.array([False, True], dtype=bool),
            "truncated": np.array([False, False], dtype=bool),
            "qpos": np.array([[1.0, 2.0, 0.0], [1.5, 2.5, 0.1]], dtype=np.float32),
            "qvel": np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=np.float32),
            "goal": np.array([[4.0, 5.0], [4.0, 5.0]], dtype=np.float32),
            "annotation": {
                "agent_xy": np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32),
                "heading": np.array([0.0, 0.5], dtype=np.float32),
                "goal_xy": np.array([[4.0, 5.0], [4.0, 5.0]], dtype=np.float32),
                "relative_goal": np.array([[3.0, 3.0], [2.5, 2.5]], dtype=np.float32),
            },
        },
        {
            "action": np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
            "reward": np.array([0.0, -1.0, 3.0], dtype=np.float32),
            "terminated": np.array([False, False, False], dtype=bool),
            "truncated": np.array([False, False, True], dtype=bool),
            "qpos": np.array([[2.0, 3.0, 0.2], [2.3, 3.1, 0.3], [2.8, 3.4, 0.4]], dtype=np.float32),
            "qvel": np.array([[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.3, 0.2, 0.0]], dtype=np.float32),
            "goal": np.array([[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]], dtype=np.float32),
            "annotation": {
                "agent_xy": np.array([[2.0, 3.0], [2.3, 3.1], [2.8, 3.4]], dtype=np.float32),
                "heading": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "goal_xy": np.array([[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]], dtype=np.float32),
                "relative_goal": np.array([[4.0, 4.0], [3.7, 3.9], [3.2, 3.6]], dtype=np.float32),
            },
        },
    ]

    for episode in episodes:
        length = int(episode["reward"].shape[0])
        achieved_goal = np.asarray(episode["qpos"][:, :2], dtype=np.float32)
        episode["obs"] = {
            "observation": np.asarray(episode["qvel"], dtype=np.float32),
            "start_pos": np.repeat(achieved_goal[:1], length, axis=0),
            "achieved_goal": achieved_goal,
            "desired_goal": np.asarray(episode["goal"], dtype=np.float32),
        }
        episode["summary"] = {
            "episode_length": length,
            "return": float(np.asarray(episode["reward"], dtype=np.float64).sum()),
            "terminated_reason": "terminated" if bool(episode["terminated"][-1]) else "truncated",
            "path_length": 1.0,
            "net_displacement": 1.0,
            "coverage_score": 1.0,
            "stuck_ratio": 0.0,
            "collision_ratio": 0.0,
            "goal_reached": bool(episode["terminated"][-1]),
            "mean_speed": 0.5,
            "mean_turn_rate": 0.0,
        }
    return episodes


def test_write_shard_writes_offsets_steps_annotations_and_attrs(tmp_path: Path):
    episodes = _synthetic_episodes()
    dataset_meta = {"dataset_name": "pointmaze-mini", "dataset_version": "v1", "seed": 123}

    shard_path = write_shard(
        output_dir=tmp_path,
        shard_id=3,
        episodes=episodes,
        dataset_meta=dataset_meta,
        storage_format="zarr",
    )

    assert shard_path.exists()

    root = zarr.open_group(str(shard_path), mode="r")
    np.testing.assert_array_equal(root["episode_lengths"][:], np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(root["episode_offsets"][:], np.array([0, 2], dtype=np.int64))

    assert root["step/action"].shape == (5, 2)
    assert root["step/reward"].shape == (5,)
    assert root["step/qpos"].shape == (5, 3)
    assert root["step/qvel"].shape == (5, 3)
    assert root["step/goal"].shape == (5, 2)
    assert root["step/terminated"].shape == (5,)
    assert root["step/truncated"].shape == (5,)
    assert root["obs/observation"].shape == (5, 3)
    assert root["obs/start_pos"].shape == (5, 2)
    assert root["obs/achieved_goal"].shape == (5, 2)
    assert root["obs/desired_goal"].shape == (5, 2)
    np.testing.assert_array_equal(root["obs/achieved_goal"][:], np.vstack([ep["obs"]["achieved_goal"] for ep in episodes]))

    np.testing.assert_array_equal(root["annotation/agent_xy"][:], np.vstack([ep["annotation"]["agent_xy"] for ep in episodes]))
    np.testing.assert_array_equal(root["annotation/heading"][:], np.concatenate([ep["annotation"]["heading"] for ep in episodes]))
    np.testing.assert_array_equal(root["annotation/goal_xy"][:], np.vstack([ep["annotation"]["goal_xy"] for ep in episodes]))
    np.testing.assert_array_equal(
        root["annotation/relative_goal"][:],
        np.vstack([ep["annotation"]["relative_goal"] for ep in episodes]),
    )

    for key, value in dataset_meta.items():
        assert root.attrs[key] == value

    summaries = json.loads(root.attrs["episode_summaries_json"])
    assert len(summaries) == 2
    assert summaries[0]["episode_length"] == 2
    assert summaries[0]["return"] == 3.0
    assert summaries[0]["terminated_reason"] == "terminated"
    assert summaries[0]["mean_speed"] == 0.5
    assert summaries[1]["episode_length"] == 3
    assert summaries[1]["return"] == 2.0
    assert summaries[1]["episode_id_in_shard"] == 1


def test_write_shard_can_write_compact_npz(tmp_path: Path):
    episodes = _synthetic_episodes()
    dataset_meta = {"dataset_name": "pointmaze-mini", "dataset_schema": "pointmaze_mujoco_zarr", "dataset_schema_version": 1}

    shard_path = write_shard(
        output_dir=tmp_path,
        shard_id=2,
        episodes=episodes,
        dataset_meta=dataset_meta,
        storage_format="npz",
    )

    assert shard_path.name == "shard_000002.npz"
    with np.load(shard_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["episode_lengths"], np.array([2, 3], dtype=np.int64))
        assert data["step/action"].shape == (5, 2)
        assert data["obs/achieved_goal"].shape == (5, 2)
        stored_meta = json.loads(str(data["dataset_meta_json"]))
        assert stored_meta["dataset_name"] == "pointmaze-mini"
        summaries = json.loads(str(data["episode_summaries_json"]))
        assert summaries[1]["episode_id_in_shard"] == 1


def test_write_dataset_metadata_writes_json(tmp_path: Path):
    dataset_meta = {
        "dataset_name": "pointmaze-mini",
        "dataset_version": "v1",
        "seed": 123,
        "num_episodes": 2,
    }

    metadata_path = write_dataset_metadata(tmp_path, dataset_meta)
    assert metadata_path.exists()
    assert metadata_path.name == "dataset_metadata.json"

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload == dataset_meta


def test_write_shard_rejects_empty_episodes(tmp_path: Path):
    with pytest.raises(ValueError, match="non-empty"):
        write_shard(output_dir=tmp_path, shard_id=0, episodes=[], dataset_meta={}, storage_format="zarr")


def test_write_shard_validates_timestep_alignment(tmp_path: Path):
    episodes = _synthetic_episodes()
    episodes[0]["qvel"] = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="field 'qvel'"):
        write_shard(output_dir=tmp_path, shard_id=0, episodes=episodes, dataset_meta={}, storage_format="zarr")


def test_write_shard_validates_obs_alignment(tmp_path: Path):
    episodes = _synthetic_episodes()
    episodes[0]["obs"]["achieved_goal"] = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="field 'obs/achieved_goal'"):
        write_shard(output_dir=tmp_path, shard_id=0, episodes=episodes, dataset_meta={}, storage_format="zarr")


def test_write_shard_normalizes_non_scalar_metadata_attrs(tmp_path: Path):
    episodes = _synthetic_episodes()
    dataset_meta = {
        "dataset_name": "pointmaze-mini",
        "seed": np.int64(7),
        "data_root": Path("/tmp/example"),
        "nested": {"a": 1},
        "array_meta": np.array([[1, 2], [3, 4]], dtype=np.int64),
    }

    shard_path = write_shard(output_dir=tmp_path, shard_id=0, episodes=episodes, dataset_meta=dataset_meta, storage_format="zarr")
    root = zarr.open_group(str(shard_path), mode="r")

    assert root.attrs["seed"] == 7
    assert root.attrs["data_root"] == "/tmp/example"
    assert "nested" not in root.attrs
    assert "array_meta" not in root.attrs

    stored_meta = json.loads(root.attrs["dataset_meta_json"])
    assert stored_meta["seed"] == 7
    assert stored_meta["data_root"] == "/tmp/example"
    assert stored_meta["nested"] == {"a": 1}
    assert stored_meta["array_meta"] == [[1, 2], [3, 4]]

    metadata_path = write_dataset_metadata(tmp_path, dataset_meta)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["array_meta"] == [[1, 2], [3, 4]]


def test_write_shard_ignores_reserved_scalar_attr_collisions(tmp_path: Path):
    episodes = _synthetic_episodes()
    dataset_meta = {
        "dataset_name": "pointmaze-mini",
        "dataset_meta_json": "user-overwrite-attempt",
        "episode_summaries_json": "user-overwrite-attempt",
    }

    shard_path = write_shard(output_dir=tmp_path, shard_id=0, episodes=episodes, dataset_meta=dataset_meta, storage_format="zarr")
    root = zarr.open_group(str(shard_path), mode="r")

    stored_meta = json.loads(root.attrs["dataset_meta_json"])
    assert stored_meta["dataset_meta_json"] == "user-overwrite-attempt"
    assert stored_meta["episode_summaries_json"] == "user-overwrite-attempt"

    summaries = json.loads(root.attrs["episode_summaries_json"])
    assert isinstance(summaries, list)
    assert len(summaries) == 2


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_write_metadata_rejects_non_finite_float(tmp_path: Path, bad_value: float):
    episodes = _synthetic_episodes()
    dataset_meta = {"dataset_name": "pointmaze-mini", "bad": bad_value}

    with pytest.raises(ValueError, match="non-finite"):
        write_shard(output_dir=tmp_path, shard_id=0, episodes=episodes, dataset_meta=dataset_meta, storage_format="zarr")

    with pytest.raises(ValueError, match="non-finite"):
        write_dataset_metadata(tmp_path, dataset_meta)
