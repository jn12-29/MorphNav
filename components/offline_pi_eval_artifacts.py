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
import torch as th

from components.dataset_gen.pointmaze_config import POINTMAZE_POLICY_OBS_KEYS
from components.offline_pi_rehearsal import decode_offline_pi_coordinates, load_offline_pi_batches
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


def _policy_lstm_shape(policy: PathIntegrationRecurrentActorCriticPolicy) -> tuple[int, int]:
    n_lstm_layers, _n_envs, lstm_hidden_size = policy.lstm_hidden_state_shape
    return int(n_lstm_layers), int(lstm_hidden_size)


def _resolve_bounds(target_xy: np.ndarray, bounds: tuple[float, float, float, float] | None = None) -> np.ndarray:
    if bounds is not None:
        return np.asarray(bounds, dtype=np.float32)
    if target_xy.size == 0:
        return np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    min_x = float(np.min(target_xy[:, 0]))
    max_x = float(np.max(target_xy[:, 0]))
    min_y = float(np.min(target_xy[:, 1]))
    max_y = float(np.max(target_xy[:, 1]))
    pad_x = max((max_x - min_x) * 0.05, 1e-3)
    pad_y = max((max_y - min_y) * 0.05, 1e-3)
    return np.asarray([min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y], dtype=np.float32)


def collect_probe_predictions(
    model: Any,
    dataset_root: str | Path,
    *,
    batch_size_sequences: int,
    max_seq_len: int | None,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    policy: PathIntegrationRecurrentActorCriticPolicy = model.policy
    policy.set_training_mode(False)
    n_lstm_layers, lstm_hidden_size = _policy_lstm_shape(policy)

    pred_parts = []
    target_parts = []
    mask_parts = []
    with th.no_grad():
        for batch in load_offline_pi_batches(
            dataset_root,
            batch_size_sequences=batch_size_sequences,
            max_seq_len=max_seq_len,
            shuffle=False,
            device=policy.device,
            obs_keys=POINTMAZE_POLICY_OBS_KEYS,
            target_key="achieved_goal",
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
        ):
            outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
            decoded = decode_offline_pi_coordinates(policy, outputs.pc_logits, batch.target_pos, batch.mask)
            mask = decoded.mask.detach().cpu().numpy().astype(bool)
            pred_parts.append(decoded.pred_xy.detach().cpu().numpy())
            target_parts.append(decoded.target_xy.detach().cpu().numpy())
            mask_parts.append(mask)
            if max_steps is not None and sum(part.shape[0] for part in pred_parts) >= max_steps:
                break

    if not pred_parts:
        raise ValueError("No probe predictions were collected")
    pred_xy = np.concatenate(pred_parts, axis=0)
    target_xy = np.concatenate(target_parts, axis=0)
    mask = np.concatenate(mask_parts, axis=0)
    if max_steps is not None:
        pred_xy = pred_xy[:max_steps]
        target_xy = target_xy[:max_steps]
        mask = mask[:max_steps]
    diff = pred_xy - target_xy
    return {
        "pred_xy": pred_xy.astype(np.float32),
        "target_xy": target_xy.astype(np.float32),
        "mask": mask.astype(bool),
        "squared_error": diff.astype(np.float32) ** 2,
        "absolute_error": np.abs(diff).astype(np.float32),
    }


def _error_summary(pred_xy: np.ndarray, target_xy: np.ndarray, bounds: np.ndarray) -> dict[str, Any]:
    diff = pred_xy - target_xy
    squared = diff**2
    abs_diff = np.abs(diff)
    euclidean = np.linalg.norm(diff, axis=1)
    mse = float(np.mean(squared.mean(axis=1))) if pred_xy.size else 0.0
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)) if pred_xy.size else 0.0,
        "x_mae": float(np.mean(abs_diff[:, 0])) if pred_xy.size else 0.0,
        "y_mae": float(np.mean(abs_diff[:, 1])) if pred_xy.size else 0.0,
        "max_error": float(np.max(euclidean)) if pred_xy.size else 0.0,
        "p50_error": float(np.percentile(euclidean, 50)) if pred_xy.size else 0.0,
        "p90_error": float(np.percentile(euclidean, 90)) if pred_xy.size else 0.0,
        "p95_error": float(np.percentile(euclidean, 95)) if pred_xy.size else 0.0,
        "num_steps": int(pred_xy.shape[0]),
        "bounds": [float(value) for value in bounds],
    }


def _plot_scatter(pred_xy: np.ndarray, target_xy: np.ndarray, bounds: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target_xy[:, 0], target_xy[:, 1], s=4, alpha=0.35, label="target")
    ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=4, alpha=0.35, label="pred")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_hist(pred_xy: np.ndarray, target_xy: np.ndarray, output_path: Path) -> None:
    euclidean = np.linalg.norm(pred_xy - target_xy, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(euclidean, bins=50)
    ax.set_xlabel("Euclidean error")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_spatial_heatmap(pred_xy: np.ndarray, target_xy: np.ndarray, bounds: np.ndarray, output_path: Path) -> None:
    euclidean = np.linalg.norm(pred_xy - target_xy, axis=1)
    hist_sum, x_edges, y_edges = np.histogram2d(
        target_xy[:, 0],
        target_xy[:, 1],
        bins=32,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        weights=euclidean,
    )
    hist_count, _, _ = np.histogram2d(
        target_xy[:, 0],
        target_xy[:, 1],
        bins=[x_edges, y_edges],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_error = hist_sum / hist_count
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        mean_error.T,
        origin="lower",
        extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("target x")
    ax.set_ylabel("target y")
    fig.colorbar(image, ax=ax, label="mean error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_probe_artifacts(
    model: Any,
    dataset_root: str | Path,
    output_dir: Path,
    *,
    epoch: int,
    batch_size_sequences: int,
    max_seq_len: int | None,
    bounds: tuple[float, float, float, float] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"epoch_{epoch:04d}"
    arrays = collect_probe_predictions(
        model,
        dataset_root,
        batch_size_sequences=batch_size_sequences,
        max_seq_len=max_seq_len,
        max_steps=max_steps,
    )
    valid_pred_xy = arrays["pred_xy"][arrays["mask"]]
    valid_target_xy = arrays["target_xy"][arrays["mask"]]
    resolved_bounds = _resolve_bounds(valid_target_xy, bounds)
    summary = _error_summary(valid_pred_xy, valid_target_xy, resolved_bounds)

    np.savez_compressed(
        output_dir / f"pred_vs_target_{suffix}.npz",
        pred_xy=arrays["pred_xy"],
        target_xy=arrays["target_xy"],
        mask=arrays["mask"],
        squared_error=arrays["squared_error"],
        absolute_error=arrays["absolute_error"],
        bounds=resolved_bounds,
    )
    (output_dir / f"error_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_scatter(valid_pred_xy, valid_target_xy, resolved_bounds, output_dir / f"coord_scatter_{suffix}.png")
    _plot_hist(valid_pred_xy, valid_target_xy, output_dir / f"error_hist_{suffix}.png")
    _plot_spatial_heatmap(valid_pred_xy, valid_target_xy, resolved_bounds, output_dir / f"spatial_error_heatmap_{suffix}.png")
    return summary
