from __future__ import annotations

from typing import Any

import numpy as np


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def summarize_episode(
    qpos: np.ndarray,
    qvel: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    info_list: list[dict[str, Any]],
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)

    length = int(rewards.shape[0])
    if terminated.shape[0] != length:
        raise ValueError(
            f"Length mismatch: rewards has {length} steps but terminated has {terminated.shape[0]} steps"
        )
    if truncated.shape[0] != length:
        raise ValueError(
            f"Length mismatch: rewards has {length} steps but truncated has {truncated.shape[0]} steps"
        )
    if len(info_list) != length:
        raise ValueError(
            f"Length mismatch: rewards has {length} steps but info_list has {len(info_list)} steps"
        )

    if qpos.ndim != 2:
        raise ValueError(f"qpos must be a 2D array with per-step rows, got shape {qpos.shape}")
    if qvel.ndim != 2:
        raise ValueError(f"qvel must be a 2D array with per-step rows, got shape {qvel.shape}")
    if qpos.shape[0] != length:
        raise ValueError(
            f"Length mismatch: rewards has {length} steps but qpos has {qpos.shape[0]} rows"
        )
    if qvel.shape[0] != length:
        raise ValueError(
            f"Length mismatch: rewards has {length} steps but qvel has {qvel.shape[0]} rows"
        )

    agent_xy = qpos[:, :2] if qpos.ndim == 2 and qpos.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    deltas = np.diff(agent_xy, axis=0) if agent_xy.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
    step_lengths = np.linalg.norm(deltas, axis=1) if deltas.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    if length == 0:
        return {
            "episode_length": 0,
            "return": 0.0,
            "terminated_reason": "unknown",
            "path_length": 0.0,
            "net_displacement": 0.0,
            "coverage_score": 0.0,
            "stuck_ratio": 0.0,
            "collision_ratio": 0.0,
            "goal_reached": False,
            "mean_speed": 0.0,
            "mean_turn_rate": 0.0,
        }

    terminated_reason = "terminated" if bool(terminated[-1]) else ("truncated" if bool(truncated[-1]) else "unknown")
    path_length = float(step_lengths.sum()) if step_lengths.size > 0 else 0.0
    net_displacement = (
        float(np.linalg.norm(agent_xy[-1] - agent_xy[0]))
        if agent_xy.shape[0] > 1
        else 0.0
    )
    coverage_score = float(len(np.unique(np.floor(agent_xy), axis=0))) if agent_xy.shape[0] > 0 else 0.0

    speeds = np.linalg.norm(qvel[:, :2], axis=1) if qvel.ndim == 2 and qvel.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    mean_speed = float(speeds.mean()) if speeds.size > 0 else 0.0
    stuck_ratio = float((speeds < 1e-6).mean()) if speeds.size > 0 else 0.0

    collisions = np.asarray(
        [
            bool(step_info.get("collision", False))
            or bool(np.any(np.asarray(step_info.get("sensordata", []), dtype=np.float32) > 0.0))
            for step_info in info_list
        ],
        dtype=bool,
    )
    collision_ratio = float(collisions.mean()) if collisions.size > 0 else 0.0
    goal_reached = bool(info_list[-1].get("success", False)) if info_list else False

    if deltas.shape[0] <= 1:
        mean_turn_rate = 0.0
    else:
        heading = np.arctan2(deltas[:, 1], deltas[:, 0])
        heading_delta = _wrap_to_pi(np.diff(heading))
        mean_turn_rate = float(np.abs(heading_delta).mean()) if heading_delta.size > 0 else 0.0

    return {
        "episode_length": length,
        "return": float(rewards.sum()),
        "terminated_reason": terminated_reason,
        "path_length": path_length,
        "net_displacement": net_displacement,
        "coverage_score": coverage_score,
        "stuck_ratio": stuck_ratio,
        "collision_ratio": collision_ratio,
        "goal_reached": goal_reached,
        "mean_speed": mean_speed,
        "mean_turn_rate": mean_turn_rate,
    }
