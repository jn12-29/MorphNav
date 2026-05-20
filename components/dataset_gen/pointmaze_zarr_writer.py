"""Shard writer for PointMaze offline PI datasets.

Writes the stable `step/*`, `annotation/*`, and policy `obs/*` arrays plus
schema metadata and complete episode summaries. Raw env `info` payloads are not
serialized in Phase 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

try:
    from .pointmaze_config import POINTMAZE_POLICY_OBS_KEYS
except ImportError:  # pragma: no cover - supports direct importlib loading in tests.
    from components.dataset_gen.pointmaze_config import POINTMAZE_POLICY_OBS_KEYS

_RESERVED_ROOT_ATTRS = frozenset({"dataset_meta_json", "episode_summaries_json"})
_SHARD_ARRAY_NAMES = (
    "episode_lengths",
    "episode_offsets",
    "step/action",
    "step/reward",
    "step/terminated",
    "step/truncated",
    "step/qpos",
    "step/qvel",
    "step/goal",
    "annotation/agent_xy",
    "annotation/heading",
    "annotation/goal_xy",
    "annotation/relative_goal",
)


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata contains non-finite float value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return [_to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    return value


def _is_json_safe_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return False


def _validate_episode_alignment(episode: dict[str, Any], episode_idx: int) -> int:
    reward = np.asarray(episode["reward"])
    length = int(reward.shape[0])

    fields = {
        "action": episode["action"],
        "terminated": episode["terminated"],
        "truncated": episode["truncated"],
        "qpos": episode["qpos"],
        "qvel": episode["qvel"],
        "goal": episode["goal"],
        "annotation/agent_xy": episode["annotation"]["agent_xy"],
        "annotation/heading": episode["annotation"]["heading"],
        "annotation/goal_xy": episode["annotation"]["goal_xy"],
        "annotation/relative_goal": episode["annotation"]["relative_goal"],
    }

    obs = episode.get("obs")
    if not isinstance(obs, dict):
        raise KeyError(f"Episode {episode_idx} is missing required 'obs' dict")
    for key in POINTMAZE_POLICY_OBS_KEYS:
        if key not in obs:
            raise KeyError(f"Episode {episode_idx} is missing required obs key {key!r}")
        fields[f"obs/{key}"] = obs[key]

    for field_name, field_value in fields.items():
        field_arr = np.asarray(field_value)
        if field_arr.shape[0] != length:
            raise ValueError(
                f"Episode {episode_idx} has timestep mismatch for field '{field_name}': "
                f"expected {length}, got {field_arr.shape[0]}"
            )

    return length


def _stack_episode_field(episodes: list[dict[str, Any]], key: str) -> np.ndarray:
    parts = [np.asarray(ep[key]) for ep in episodes]
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _stack_annotation_field(episodes: list[dict[str, Any]], key: str) -> np.ndarray:
    parts = [np.asarray(ep["annotation"][key]) for ep in episodes]
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _stack_obs_field(episodes: list[dict[str, Any]], key: str) -> np.ndarray:
    parts = [np.asarray(ep["obs"][key], dtype=np.float32) for ep in episodes]
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _build_shard_payload(episodes: list[dict], dataset_meta: dict) -> tuple[dict[str, np.ndarray], dict, list[dict]]:
    if not episodes:
        raise ValueError("episodes must be a non-empty list")

    lengths = [_validate_episode_alignment(ep, idx) for idx, ep in enumerate(episodes)]
    episode_lengths = np.asarray(lengths, dtype=np.int64)
    episode_offsets = np.zeros_like(episode_lengths)
    if episode_lengths.size > 1:
        episode_offsets[1:] = np.cumsum(episode_lengths[:-1], dtype=np.int64)

    step_action = _stack_episode_field(episodes, "action")
    step_reward = _stack_episode_field(episodes, "reward")
    step_terminated = _stack_episode_field(episodes, "terminated").astype(bool, copy=False)
    step_truncated = _stack_episode_field(episodes, "truncated").astype(bool, copy=False)
    step_qpos = _stack_episode_field(episodes, "qpos")
    step_qvel = _stack_episode_field(episodes, "qvel")
    step_goal = _stack_episode_field(episodes, "goal")

    obs_arrays = {key: _stack_obs_field(episodes, key) for key in POINTMAZE_POLICY_OBS_KEYS}

    ann_agent_xy = _stack_annotation_field(episodes, "agent_xy")
    ann_heading = _stack_annotation_field(episodes, "heading")
    ann_goal_xy = _stack_annotation_field(episodes, "goal_xy")
    ann_relative_goal = _stack_annotation_field(episodes, "relative_goal")

    summaries = []
    for idx, ep in enumerate(episodes):
        if "summary" not in ep:
            raise KeyError(f"Episode {idx} is missing required 'summary'")
        summary = dict(ep["summary"])
        summary["episode_id_in_shard"] = idx
        summaries.append(summary)

    normalized_meta = _to_json_compatible(dataset_meta)
    arrays = {
        "episode_lengths": episode_lengths,
        "episode_offsets": episode_offsets,
        "step/action": step_action,
        "step/reward": step_reward,
        "step/terminated": step_terminated,
        "step/truncated": step_truncated,
        "step/qpos": step_qpos,
        "step/qvel": step_qvel,
        "step/goal": step_goal,
        "annotation/agent_xy": ann_agent_xy,
        "annotation/heading": ann_heading,
        "annotation/goal_xy": ann_goal_xy,
        "annotation/relative_goal": ann_relative_goal,
    }
    for key, value in obs_arrays.items():
        arrays[f"obs/{key}"] = value
    return arrays, normalized_meta, summaries


def _write_zarr_shard(output_dir: Path, shard_id: int, arrays: dict[str, np.ndarray], normalized_meta: dict, summaries: list[dict]) -> Path:
    shard_path = output_dir / f"shard_{shard_id:06d}.zarr"
    root = zarr.open_group(str(shard_path), mode="w")
    for key in _SHARD_ARRAY_NAMES:
        root.create_array(key, data=arrays[key])
    for key in POINTMAZE_POLICY_OBS_KEYS:
        root.create_array(f"obs/{key}", data=arrays[f"obs/{key}"])

    root.attrs["dataset_meta_json"] = json.dumps(normalized_meta, sort_keys=True, allow_nan=False)
    for key, value in normalized_meta.items():
        key_str = str(key)
        if key_str in _RESERVED_ROOT_ATTRS:
            continue
        if _is_json_safe_scalar(value):
            root.attrs[key_str] = value
    root.attrs["episode_summaries_json"] = json.dumps(summaries, sort_keys=True, allow_nan=False)

    return shard_path


def _write_npz_shard(output_dir: Path, shard_id: int, arrays: dict[str, np.ndarray], normalized_meta: dict, summaries: list[dict]) -> Path:
    shard_path = output_dir / f"shard_{shard_id:06d}.npz"
    payload = dict(arrays)
    payload["dataset_meta_json"] = np.asarray(json.dumps(normalized_meta, sort_keys=True, allow_nan=False))
    payload["episode_summaries_json"] = np.asarray(json.dumps(summaries, sort_keys=True, allow_nan=False))
    np.savez(file=str(shard_path), **payload)  # type: ignore[arg-type]
    return shard_path


def write_shard(
    output_dir: Path,
    shard_id: int,
    episodes: list[dict],
    dataset_meta: dict,
    *,
    storage_format: str = "npz",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, normalized_meta, summaries = _build_shard_payload(episodes, dataset_meta)
    if storage_format == "zarr":
        return _write_zarr_shard(output_dir, shard_id, arrays, normalized_meta, summaries)
    if storage_format == "npz":
        return _write_npz_shard(output_dir, shard_id, arrays, normalized_meta, summaries)
    raise ValueError("storage_format must be 'zarr' or 'npz'")


def write_dataset_metadata(output_dir: Path, dataset_meta: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_metadata.json"
    normalized_meta = _to_json_compatible(dataset_meta)
    path.write_text(
        json.dumps(normalized_meta, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
