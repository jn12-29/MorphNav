from __future__ import annotations

import numpy as np


def _compute_heading(agent_xy: np.ndarray) -> np.ndarray:
    deltas = np.diff(agent_xy, axis=0, prepend=agent_xy[:1])
    heading = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
    if heading.shape[0] <= 1:
        return heading

    delta_norm = np.linalg.norm(deltas, axis=1)
    valid_indices = np.flatnonzero(delta_norm > 0.0)
    if valid_indices.size > 0:
        heading[0] = heading[int(valid_indices[0])]
    return heading


def annotate_episode(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    agent_xy = np.asarray(episode["qpos"], dtype=np.float32)[..., :2]
    goal = np.asarray(episode["goal"], dtype=np.float32)
    if goal.ndim == 1:
        goal_xy = np.repeat(goal[:2][None, :], agent_xy.shape[0], axis=0)
    else:
        goal_xy = goal[..., :2]
    relative_goal = goal_xy - agent_xy

    return {
        "agent_xy": agent_xy,
        "heading": _compute_heading(agent_xy),
        "goal_xy": goal_xy,
        "relative_goal": relative_goal.astype(np.float32),
    }
