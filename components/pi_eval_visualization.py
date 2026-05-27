from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphnav_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from components.offline_pi_gridscore import (
    analyze_grid_scores,
    compute_spatial_ratemaps,
    plot_ratemap_grid,
    plot_top_grid_cells,
    resolve_gridscore_bounds,
    summarize_grid_scores,
)


def _num_envs_from_obs(obs: dict[str, np.ndarray]) -> int:
    first_value = next(iter(obs.values()))
    return int(np.asarray(first_value).shape[0])


def _target_xy_from_obs(obs: dict[str, np.ndarray], target_key: str, n_envs: int) -> np.ndarray:
    if target_key not in obs:
        raise KeyError(f"PI eval visualization target key {target_key!r} is missing from observations")
    target = np.asarray(obs[target_key], dtype=np.float32).reshape(n_envs, -1)
    if target.shape[1] < 2:
        raise ValueError(f"obs[{target_key!r}] must expose at least 2 coordinates, got shape {target.shape}")
    return target[:, :2]


def _decode_pc_logits(policy: Any, pc_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(pc_logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    centers = policy.path_integration_target_encoder.centers.detach().cpu().numpy().astype(np.float32)
    return probs @ centers


def _empty_episode() -> dict[str, list[np.ndarray]]:
    return {"target_xy": [], "pred_xy": [], "bottleneck": []}


def _finalize_episode(episode: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray] | None:
    if not episode["target_xy"]:
        return None
    return {
        "target_xy": np.stack(episode["target_xy"], axis=0).astype(np.float32),
        "pred_xy": np.stack(episode["pred_xy"], axis=0).astype(np.float32),
        "bottleneck": np.stack(episode["bottleneck"], axis=0).astype(np.float32),
    }


def collect_online_pi_eval(
    model: Any,
    eval_env: Any,
    *,
    n_eval_episodes: int,
    target_key: str = "achieved_goal",
    deterministic: bool = True,
    max_total_steps: int | None = None,
) -> list[dict[str, np.ndarray]]:
    if n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be positive")
    predict_with_pi = getattr(model, "predict_with_pi", None)
    if not callable(predict_with_pi):
        raise TypeError("model must expose predict_with_pi() for PI eval visualization")

    policy = model.policy
    was_training = bool(getattr(policy, "training", False))
    set_training_mode = getattr(policy, "set_training_mode", None)
    if callable(set_training_mode):
        set_training_mode(False)

    try:
        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if not isinstance(obs, dict):
            raise TypeError("PI eval visualization requires dict observations")

        n_envs = _num_envs_from_obs(obs)
        episode_start = np.ones((n_envs,), dtype=bool)
        state = None
        current = [_empty_episode() for _ in range(n_envs)]
        completed: list[dict[str, np.ndarray]] = []
        max_steps = max_total_steps or max(1, n_eval_episodes * 1000)

        for _ in range(max_steps):
            if len(completed) >= n_eval_episodes:
                break

            actions, state, pi_outputs = predict_with_pi(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            target_xy = _target_xy_from_obs(obs, target_key, n_envs)
            pred_xy = _decode_pc_logits(policy, pi_outputs["pc_logits"]).reshape(n_envs, 2)
            bottleneck = np.asarray(pi_outputs["bottleneck"], dtype=np.float32).reshape(n_envs, -1)

            for env_idx in range(n_envs):
                current[env_idx]["target_xy"].append(target_xy[env_idx].copy())
                current[env_idx]["pred_xy"].append(pred_xy[env_idx].copy())
                current[env_idx]["bottleneck"].append(bottleneck[env_idx].copy())

            obs, _rewards, dones, _infos = eval_env.step(actions)
            dones = np.asarray(dones, dtype=bool).reshape(n_envs)
            episode_start = dones
            for env_idx, done in enumerate(dones):
                if not done:
                    continue
                episode = _finalize_episode(current[env_idx])
                if episode is not None:
                    completed.append(episode)
                current[env_idx] = _empty_episode()

        for episode in current:
            if len(completed) >= n_eval_episodes:
                break
            finalized = _finalize_episode(episode)
            if finalized is not None:
                completed.append(finalized)

        if not completed:
            raise ValueError("No PI eval trajectories were collected")
        return completed[:n_eval_episodes]
    finally:
        if callable(set_training_mode):
            set_training_mode(was_training)


def _flatten_episodes(episodes: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_xy = np.concatenate([episode["target_xy"] for episode in episodes], axis=0)
    pred_xy = np.concatenate([episode["pred_xy"] for episode in episodes], axis=0)
    bottleneck = np.concatenate([episode["bottleneck"] for episode in episodes], axis=0)
    return target_xy, pred_xy, bottleneck


def _localization_summary(pred_xy: np.ndarray, target_xy: np.ndarray) -> dict[str, float | int]:
    diff = pred_xy - target_xy
    squared = diff**2
    abs_diff = np.abs(diff)
    euclidean = np.linalg.norm(diff, axis=1)
    mse = float(np.mean(squared.mean(axis=1))) if pred_xy.size else float("nan")
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if pred_xy.size else float("nan"),
        "mae": float(np.mean(abs_diff)) if pred_xy.size else float("nan"),
        "x_mae": float(np.mean(abs_diff[:, 0])) if pred_xy.size else float("nan"),
        "y_mae": float(np.mean(abs_diff[:, 1])) if pred_xy.size else float("nan"),
        "max_error": float(np.max(euclidean)) if pred_xy.size else float("nan"),
        "p50_error": float(np.percentile(euclidean, 50)) if pred_xy.size else float("nan"),
        "p90_error": float(np.percentile(euclidean, 90)) if pred_xy.size else float("nan"),
        "num_steps": int(pred_xy.shape[0]),
    }


def _plot_prediction_summary(
    episodes: list[dict[str, np.ndarray]],
    pred_xy: np.ndarray,
    target_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_traj, ax_scatter, ax_hist, ax_heat = axes.flat
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(episodes), 10)))

    for idx, episode in enumerate(episodes[:4]):
        color = colors[idx % len(colors)]
        target = episode["target_xy"]
        pred = episode["pred_xy"]
        ax_traj.plot(target[:, 0], target[:, 1], color=color, linewidth=1.4, label=f"target {idx}")
        ax_traj.plot(pred[:, 0], pred[:, 1], color=color, linewidth=1.1, linestyle="--", label=f"pred {idx}")
        ax_traj.scatter(target[0, 0], target[0, 1], color=color, s=18)
    ax_traj.set_title("PI decoded trajectories")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_xlim(bounds[0], bounds[1])
    ax_traj.set_ylim(bounds[2], bounds[3])
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(fontsize=7, ncol=2)

    ax_scatter.scatter(target_xy[:, 0], target_xy[:, 1], s=5, alpha=0.35, label="target")
    ax_scatter.scatter(pred_xy[:, 0], pred_xy[:, 1], s=5, alpha=0.35, label="pred")
    ax_scatter.set_title("Target vs decoded position")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_xlim(bounds[0], bounds[1])
    ax_scatter.set_ylim(bounds[2], bounds[3])
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.legend(fontsize=8)

    euclidean = np.linalg.norm(pred_xy - target_xy, axis=1)
    ax_hist.hist(euclidean, bins=40, color="steelblue")
    ax_hist.set_title("Localization error")
    ax_hist.set_xlabel("Euclidean error")
    ax_hist.set_ylabel("count")

    hist_sum, x_edges, y_edges = np.histogram2d(
        target_xy[:, 0],
        target_xy[:, 1],
        bins=32,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        weights=euclidean,
    )
    hist_count, _, _ = np.histogram2d(target_xy[:, 0], target_xy[:, 1], bins=[x_edges, y_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_error = hist_sum / hist_count
    image = ax_heat.imshow(
        mean_error.T,
        origin="lower",
        extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
        aspect="auto",
        cmap="magma",
    )
    ax_heat.set_title("Mean error by target position")
    ax_heat.set_xlabel("target x")
    ax_heat.set_ylabel("target y")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_online_pi_eval_visualization(
    model: Any,
    eval_env: Any,
    output_dir: Path,
    *,
    n_eval_episodes: int = 4,
    target_key: str = "achieved_goal",
    deterministic: bool = True,
    n_bins: int = 32,
    bounds: tuple[float, float, float, float] | None = None,
    max_units: int | None = None,
    top_k: int = 8,
    max_total_steps: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_online_pi_eval(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        target_key=target_key,
        deterministic=deterministic,
        max_total_steps=max_total_steps,
    )
    target_xy, pred_xy, bottleneck = _flatten_episodes(episodes)
    activations = bottleneck if max_units is None else bottleneck[:, :max_units]
    resolved_bounds = resolve_gridscore_bounds(target_xy, bounds)
    ratemaps = compute_spatial_ratemaps(target_xy, activations, n_bins=n_bins, bounds=resolved_bounds)
    autocorrs, grid_scores = analyze_grid_scores(ratemaps)
    grid_summary = summarize_grid_scores(
        target_xy,
        grid_scores,
        bounds=resolved_bounds,
        n_bins=n_bins,
        max_steps=None,
        max_units=max_units,
        top_k=top_k,
    )
    summary = {
        "num_episodes": int(len(episodes)),
        "episode_lengths": [int(episode["target_xy"].shape[0]) for episode in episodes],
        "num_steps": int(target_xy.shape[0]),
        "target_key": target_key,
        "bounds": [float(value) for value in resolved_bounds],
        "localization": _localization_summary(pred_xy, target_xy),
        "gridscore": grid_summary,
    }

    np.savez_compressed(
        output_dir / "pi_eval_data.npz",
        target_xy=target_xy,
        pred_xy=pred_xy,
        bottleneck=activations,
        ratemaps=ratemaps,
        autocorrs=autocorrs,
        grid_scores=grid_scores,
        bounds=np.asarray(resolved_bounds, dtype=np.float32),
    )
    (output_dir / "pi_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_top_grid_cells(ratemaps, autocorrs, grid_scores, output_dir, top_k)
    plot_ratemap_grid(ratemaps, output_dir)
    _plot_prediction_summary(episodes, pred_xy, target_xy, resolved_bounds, output_dir / "trajectory_preview.png")
    return summary
