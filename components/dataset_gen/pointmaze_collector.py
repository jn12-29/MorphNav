from __future__ import annotations

from typing import Any

import numpy as np

from .pointmaze_episode import summarize_episode


def collect_episode(
    env,
    policy,
    episode_id: int,
    episode_seed: int,
    env_metadata: dict[str, Any],
    policy_metadata: dict[str, Any],
) -> dict[str, Any]:
    max_steps_raw = env_metadata.get("max_episode_steps")
    max_steps: int | None = None
    if max_steps_raw is not None:
        max_steps = int(max_steps_raw)
        if max_steps <= 0:
            raise ValueError("env_metadata.max_episode_steps must be a positive integer when provided")

    obs, info = env.reset(seed=episode_seed)

    expected_obs_keys = tuple(obs.keys())
    expected_obs_key_set = set(expected_obs_keys)
    obs_buffer: dict[str, list[np.ndarray]] = {key: [] for key in expected_obs_keys}
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []

    if "qpos" in info:
        last_xy = np.asarray(info["qpos"], dtype=np.float32)[:2]
    elif "achieved_goal" in obs:
        last_xy = np.asarray(obs["achieved_goal"], dtype=np.float32)[:2]
    else:
        raise KeyError("Missing required initial position in reset output: expected info['qpos'] or obs['achieved_goal']")
    heading = 0.0
    collision = bool(info.get("collision", False))
    done = False

    while not done:
        action = np.asarray(
            policy.act(agent_xy=last_xy, heading=heading, collision=collision),
            dtype=np.float32,
        )
        next_obs, reward, term, trunc, step_info = env.step(action)
        if "desired_goal" not in next_obs:
            raise KeyError("Missing required field in step observation: 'desired_goal'")
        next_obs_keys = tuple(next_obs.keys())
        if set(next_obs_keys) != expected_obs_key_set:
            raise ValueError(
                f"Observation keys changed across steps: expected {expected_obs_keys}, got {next_obs_keys}"
            )
        if "qpos" not in step_info:
            raise KeyError("Missing required field in step info: 'qpos'")
        if "qvel" not in step_info:
            raise KeyError("Missing required field in step info: 'qvel'")

        for key in expected_obs_keys:
            obs_buffer[key].append(np.asarray(next_obs[key], dtype=np.float32))

        step_qpos = np.asarray(step_info["qpos"], dtype=np.float32)
        step_qvel = np.asarray(step_info["qvel"], dtype=np.float32)
        goal = np.asarray(next_obs["desired_goal"], dtype=np.float32)

        actions.append(action)
        rewards.append(float(reward))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        qpos.append(step_qpos)
        qvel.append(step_qvel)
        goals.append(goal)
        infos.append(dict(step_info))

        new_xy = step_qpos[:2]
        delta = new_xy - last_xy
        if float(np.linalg.norm(delta)) > 0.0:
            heading = float(np.arctan2(delta[1], delta[0]))
        collision = bool(step_info.get("collision", False))
        last_xy = new_xy
        if max_steps is not None and len(actions) >= max_steps and not (term or trunc):
            truncated[-1] = True
            trunc = True
        done = bool(term or trunc)

    episode = {
        "episode_id": int(episode_id),
        "seed": int(episode_seed),
        "env_metadata": dict(env_metadata),
        "policy_metadata": dict(policy_metadata),
        "obs": {key: np.asarray(values, dtype=np.float32) for key, values in obs_buffer.items()},
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "qpos": np.asarray(qpos, dtype=np.float32),
        "qvel": np.asarray(qvel, dtype=np.float32),
        "goal": np.asarray(goals, dtype=np.float32),
        "info": infos,
    }
    episode["summary"] = summarize_episode(
        qpos=episode["qpos"],
        qvel=episode["qvel"],
        rewards=episode["reward"],
        terminated=episode["terminated"],
        truncated=episode["truncated"],
        info_list=infos,
    )
    return episode
