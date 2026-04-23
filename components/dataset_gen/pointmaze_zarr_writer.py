from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

_RESERVED_ROOT_ATTRS = frozenset({"dataset_meta_json", "episode_summaries_json"})


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


def write_shard(
    output_dir: Path,
    shard_id: int,
    episodes: list[dict],
    dataset_meta: dict,
) -> Path:
    if not episodes:
        raise ValueError("episodes must be a non-empty list")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{shard_id:06d}.zarr"

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

    ann_agent_xy = _stack_annotation_field(episodes, "agent_xy")
    ann_heading = _stack_annotation_field(episodes, "heading")
    ann_goal_xy = _stack_annotation_field(episodes, "goal_xy")
    ann_relative_goal = _stack_annotation_field(episodes, "relative_goal")

    root = zarr.open_group(str(shard_path), mode="w")
    root.create_array("episode_lengths", data=episode_lengths)
    root.create_array("episode_offsets", data=episode_offsets)
    root.create_array("step/action", data=step_action)
    root.create_array("step/reward", data=step_reward)
    root.create_array("step/terminated", data=step_terminated)
    root.create_array("step/truncated", data=step_truncated)
    root.create_array("step/qpos", data=step_qpos)
    root.create_array("step/qvel", data=step_qvel)
    root.create_array("step/goal", data=step_goal)
    root.create_array("annotation/agent_xy", data=ann_agent_xy)
    root.create_array("annotation/heading", data=ann_heading)
    root.create_array("annotation/goal_xy", data=ann_goal_xy)
    root.create_array("annotation/relative_goal", data=ann_relative_goal)

    summaries = []
    for idx, ep in enumerate(episodes):
        rewards = np.asarray(ep["reward"], dtype=np.float64)
        summaries.append(
            {
                "episode_id_in_shard": idx,
                "episode_length": int(rewards.shape[0]),
                "return": float(rewards.sum()),
            }
        )

    normalized_meta = _to_json_compatible(dataset_meta)
    root.attrs["dataset_meta_json"] = json.dumps(normalized_meta, sort_keys=True, allow_nan=False)
    for key, value in normalized_meta.items():
        key_str = str(key)
        if key_str in _RESERVED_ROOT_ATTRS:
            continue
        if _is_json_safe_scalar(value):
            root.attrs[key_str] = value
    root.attrs["episode_summaries_json"] = json.dumps(summaries, sort_keys=True, allow_nan=False)

    return shard_path


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
