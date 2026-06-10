from __future__ import annotations

from typing import Any

import numpy as np
import torch as th

from components.dataset_gen.pointmaze_config import POINTMAZE_POLICY_OBS_KEYS
from components.pointmaze_rendering_data import PointMazeTrajectory


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - np.max(values, axis=-1, keepdims=True)
    probs = np.exp(values)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs


def decode_pc_logits(policy: Any, pc_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(pc_logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    centers = policy.path_integration_target_encoder.centers.detach().cpu().numpy().astype(np.float32)
    return _softmax(logits) @ centers


def _policy_lstm_shape(policy: Any) -> tuple[int, int]:
    shape = getattr(policy, "lstm_hidden_state_shape", None)
    if shape is None:
        n_layers = int(getattr(getattr(policy, "lstm_actor", None), "num_layers", 1))
        hidden_size = int(getattr(policy, "lstm_output_dim", 1))
        return n_layers, hidden_size
    return int(shape[0]), int(shape[2])


def _policy_device(policy: Any) -> th.device:
    raw_device = getattr(policy, "device", "cpu")
    return th.device(raw_device)


def _chunk_obs_for_pi(obs: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    chunk = {key: np.asarray(value[start:end], dtype=np.float32).copy() for key, value in obs.items()}
    if "start_pos" in chunk and "achieved_goal" in chunk and chunk["start_pos"].shape[-1] >= 2:
        chunk["start_pos"] = np.repeat(chunk["achieved_goal"][:1, :2], end - start, axis=0).astype(np.float32)
    return chunk


def predict_dataset_episode_pi(
    model: Any,
    trajectory: PointMazeTrajectory,
    *,
    max_seq_len: int | None = 1000,
) -> PointMazeTrajectory:
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive when provided")

    policy = model.policy
    was_training = bool(getattr(policy, "training", False))
    set_training_mode = getattr(policy, "set_training_mode", None)
    if callable(set_training_mode):
        set_training_mode(False)

    pred_parts: list[np.ndarray] = []
    bottleneck_parts: list[np.ndarray] = []
    try:
        n_layers, hidden_size = _policy_lstm_shape(policy)
        device = _policy_device(policy)
        window = max_seq_len or trajectory.length
        for start in range(0, trajectory.length, window):
            end = min(start + window, trajectory.length)
            obs_np = _chunk_obs_for_pi(trajectory.obs, start, end)
            obs_th = {key: th.as_tensor(value, dtype=th.float32, device=device) for key, value in obs_np.items()}
            states = (
                th.zeros((n_layers, 1, hidden_size), dtype=th.float32, device=device),
                th.zeros((n_layers, 1, hidden_size), dtype=th.float32, device=device),
            )
            episode_starts = th.zeros((end - start,), dtype=th.float32, device=device)
            episode_starts[0] = 1.0
            with th.no_grad():
                outputs, _new_states = policy.forward_pi(obs_th, states, episode_starts)
            pred_parts.append(decode_pc_logits(policy, outputs.pc_logits.detach().cpu().numpy()))
            bottleneck = getattr(outputs, "bottleneck", None)
            if bottleneck is not None:
                bottleneck_parts.append(bottleneck.detach().cpu().numpy().astype(np.float32))
    finally:
        if callable(set_training_mode):
            set_training_mode(was_training)

    pred_xy = np.concatenate(pred_parts, axis=0).astype(np.float32)
    bottleneck = np.concatenate(bottleneck_parts, axis=0).astype(np.float32) if bottleneck_parts else None
    return trajectory.with_predictions(pred_xy, bottleneck)


def _state_template_from_env(env: Any, field: str, fallback_width: int) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    data = getattr(base, "data", None)
    value = getattr(data, field, None)
    if value is not None:
        return np.asarray(value, dtype=np.float32).copy()
    return np.zeros((fallback_width,), dtype=np.float32)


def _rollout_current_qpos(obs: dict[str, np.ndarray], info: dict[str, Any], env: Any) -> np.ndarray:
    qpos = np.asarray(info.get("qpos", _state_template_from_env(env, "qpos", 2)), dtype=np.float32).copy()
    if qpos.shape[0] < 2:
        qpos = np.pad(qpos, (0, 2 - qpos.shape[0]))
    qpos[:2] = np.asarray(obs["achieved_goal"], dtype=np.float32)[:2]
    return qpos


def _rollout_current_qvel(obs: dict[str, np.ndarray], info: dict[str, Any], env: Any) -> np.ndarray:
    qvel = np.asarray(info.get("qvel", _state_template_from_env(env, "qvel", 2)), dtype=np.float32).copy()
    observation = np.asarray(obs.get("observation", []), dtype=np.float32).reshape(-1)
    width = min(qvel.shape[0], observation.shape[0])
    if width > 0:
        qvel[:width] = observation[:width]
    return qvel


def _as_unbatched_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim > 1 and action.shape[0] == 1:
        return action[0]
    return action


def collect_recurrent_policy_rollout(
    model: Any,
    env: Any,
    *,
    steps: int = 1000,
    seed: int = 0,
    deterministic: bool = True,
) -> PointMazeTrajectory:
    if steps <= 0:
        raise ValueError("steps must be positive")
    predict_with_pi = getattr(model, "predict_with_pi", None)
    if not callable(predict_with_pi):
        raise TypeError("model must expose predict_with_pi()")

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs, info = reset_result, {}
    if not isinstance(obs, dict):
        raise TypeError("PointMaze rollout rendering requires dict observations")

    state = None
    episode_start = np.ones((1,), dtype=bool)
    obs_buffer = {key: [] for key in POINTMAZE_POLICY_OBS_KEYS}
    obs_xy: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    goal_xy: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    post_qpos: list[np.ndarray] = []
    post_qvel: list[np.ndarray] = []
    pred_xy: list[np.ndarray] = []
    bottleneck: list[np.ndarray] = []

    for step_idx in range(steps):
        for key in POINTMAZE_POLICY_OBS_KEYS:
            obs_buffer[key].append(np.asarray(obs[key], dtype=np.float32).copy())
        current_xy = np.asarray(obs["achieved_goal"], dtype=np.float32)[:2]
        obs_xy.append(current_xy.copy())
        qpos.append(_rollout_current_qpos(obs, info, env))
        qvel.append(_rollout_current_qvel(obs, info, env))
        goal_xy.append(np.asarray(obs["desired_goal"], dtype=np.float32)[:2].copy())

        action, state, pi_outputs = predict_with_pi(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        action = _as_unbatched_action(action)
        decoded = decode_pc_logits(model.policy, np.asarray(pi_outputs["pc_logits"], dtype=np.float32)).reshape(-1, 2)
        pred_xy.append(decoded[0].astype(np.float32))
        bottleneck_value = np.asarray(pi_outputs.get("bottleneck", []), dtype=np.float32)
        bottleneck.append(bottleneck_value.reshape(-1).copy())

        next_obs, reward, term, trunc, step_info = env.step(action)
        done = bool(term or trunc)
        actions.append(action.copy())
        rewards.append(float(reward))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        post_qpos.append(np.asarray(step_info.get("qpos", qpos[-1]), dtype=np.float32).copy())
        post_qvel.append(np.asarray(step_info.get("qvel", qvel[-1]), dtype=np.float32).copy())

        if done and step_idx < steps - 1:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
            episode_start = np.ones((1,), dtype=bool)
        else:
            obs, info = next_obs, step_info
            episode_start = np.asarray([done], dtype=bool)

    obs_arrays = {key: np.asarray(values, dtype=np.float32) for key, values in obs_buffer.items()}
    bottleneck_array = (
        np.stack(bottleneck, axis=0).astype(np.float32)
        if bottleneck and all(value.shape == bottleneck[0].shape for value in bottleneck)
        else None
    )
    return PointMazeTrajectory(
        kind="rollout",
        episode_id=None,
        obs=obs_arrays,
        obs_xy=np.asarray(obs_xy, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float32),
        qvel=np.asarray(qvel, dtype=np.float32),
        goal_xy=np.asarray(goal_xy, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        post_action_qpos=np.asarray(post_qpos, dtype=np.float32),
        post_action_qvel=np.asarray(post_qvel, dtype=np.float32),
        pred_xy=np.asarray(pred_xy, dtype=np.float32),
        bottleneck=bottleneck_array,
    )
