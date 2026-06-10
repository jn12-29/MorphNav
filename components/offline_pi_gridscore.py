from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphnav_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.ndimage import rotate

from components.offline_pi_rehearsal import load_offline_pi_batches
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


def _policy_lstm_shape(policy: PathIntegrationRecurrentActorCriticPolicy) -> tuple[int, int]:
    n_lstm_layers, _n_envs, lstm_hidden_size = policy.lstm_hidden_state_shape
    return int(n_lstm_layers), int(lstm_hidden_size)


def collect_bottleneck_activity(
    model: Any,
    dataset_root: str | Path,
    *,
    batch_size_sequences: int,
    max_seq_len: int | None,
    max_steps: int | None = None,
    max_units: int | None = 256,
) -> tuple[np.ndarray, np.ndarray]:
    policy: PathIntegrationRecurrentActorCriticPolicy = model.policy
    policy.set_training_mode(False)
    n_lstm_layers, lstm_hidden_size = _policy_lstm_shape(policy)

    positions = []
    activations = []
    steps_seen = 0
    with th.no_grad():
        for batch in load_offline_pi_batches(
            dataset_root,
            batch_size_sequences=batch_size_sequences,
            max_seq_len=max_seq_len,
            shuffle=False,
            device=policy.device,
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
        ):
            outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
            mask = batch.mask.bool()
            position_part = batch.target_pos[mask].detach().cpu().numpy()
            activation_part = outputs.bottleneck[mask].detach().cpu().numpy()
            positions.append(position_part)
            activations.append(activation_part)
            steps_seen += int(position_part.shape[0])
            if max_steps is not None and steps_seen >= max_steps:
                break

    if not positions:
        raise ValueError("No bottleneck activity was collected from the dataset")
    position_arr = np.concatenate(positions, axis=0)
    activation_arr = np.concatenate(activations, axis=0)
    if max_steps is not None:
        position_arr = position_arr[:max_steps]
        activation_arr = activation_arr[:max_steps]
    if max_units is not None:
        activation_arr = activation_arr[:, :max_units]
    return position_arr, activation_arr


def resolve_gridscore_bounds(
    positions: np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    margin: float = 0.1,
) -> tuple[float, float, float, float]:
    if bounds is not None:
        return tuple(float(value) for value in bounds)
    return (
        float(np.min(positions[:, 0]) - margin),
        float(np.max(positions[:, 0]) + margin),
        float(np.min(positions[:, 1]) - margin),
        float(np.max(positions[:, 1]) + margin),
    )


def compute_spatial_ratemaps(
    positions: np.ndarray,
    activations: np.ndarray,
    *,
    n_bins: int,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    range_x = [bounds[0], bounds[1]]
    range_y = [bounds[2], bounds[3]]
    occupancy, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=n_bins, range=[range_x, range_y])
    ratemaps = np.empty((activations.shape[1], n_bins, n_bins), dtype=np.float32)
    for unit in range(activations.shape[1]):
        act_sum, _, _ = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=n_bins,
            range=[range_x, range_y],
            weights=activations[:, unit],
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratemaps[unit] = act_sum / occupancy
    return ratemaps


def compute_2d_autocorrelogram(ratemap: np.ndarray) -> np.ndarray:
    n_bins_y, n_bins_x = ratemap.shape
    autocorr = np.full((2 * n_bins_y - 1, 2 * n_bins_x - 1), np.nan, dtype=np.float32)
    for shift_y in range(-n_bins_y + 1, n_bins_y):
        for shift_x in range(-n_bins_x + 1, n_bins_x):
            y1_start = max(0, -shift_y)
            y1_end = min(n_bins_y, n_bins_y - shift_y)
            x1_start = max(0, -shift_x)
            x1_end = min(n_bins_x, n_bins_x - shift_x)
            y2_start = max(0, shift_y)
            y2_end = min(n_bins_y, n_bins_y + shift_y)
            x2_start = max(0, shift_x)
            x2_end = min(n_bins_x, n_bins_x + shift_x)
            overlap_1 = ratemap[y1_start:y1_end, x1_start:x1_end]
            overlap_2 = ratemap[y2_start:y2_end, x2_start:x2_end]
            valid = ~np.isnan(overlap_1) & ~np.isnan(overlap_2)
            if np.sum(valid) < 20:
                continue
            v1 = overlap_1[valid]
            v2 = overlap_2[valid]
            denom = np.std(v1) * np.std(v2)
            if denom > 0.0:
                autocorr[shift_y + n_bins_y - 1, shift_x + n_bins_x - 1] = np.corrcoef(v1, v2)[0, 1]
            else:
                autocorr[shift_y + n_bins_y - 1, shift_x + n_bins_x - 1] = 0.0
    return autocorr


def calculate_grid_score(autocorr: np.ndarray) -> float:
    center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    y, x = np.ogrid[-center_y : center_y + 1, -center_x : center_x + 1]
    dist_from_center = np.sqrt(x**2 + y**2)
    inner_radius = 4
    max_outer_radius = min(center_y, center_x) - 2
    if max_outer_radius <= inner_radius:
        return float("nan")

    best = -2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for outer_radius in range(inner_radius + 4, max_outer_radius + 1, 2):
            ring_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
            masked = autocorr.copy()
            masked[~ring_mask] = np.nan
            correlations = []
            for angle in (30, 60, 90, 120, 150):
                rotated = rotate(masked, angle, reshape=False, order=1, cval=np.nan)
                valid = ~np.isnan(masked) & ~np.isnan(rotated)
                if np.sum(valid) < 10:
                    correlations.append(np.nan)
                    continue
                correlations.append(np.corrcoef(masked[valid], rotated[valid])[0, 1])
            if not np.any(np.isnan(correlations)):
                score = min(correlations[1], correlations[3]) - max(correlations[0], correlations[2], correlations[4])
                best = max(best, float(score))
    return best if best != -2.0 else float("nan")


def analyze_grid_scores(ratemaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    autocorrs = []
    scores = []
    for ratemap in ratemaps:
        autocorr = compute_2d_autocorrelogram(ratemap)
        autocorrs.append(autocorr)
        scores.append(calculate_grid_score(autocorr))
    return np.asarray(autocorrs), np.asarray(scores, dtype=np.float32)


def summarize_grid_scores(
    positions: np.ndarray,
    grid_scores: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    n_bins: int,
    max_steps: int | None,
    max_units: int | None,
    top_k: int,
) -> dict[str, Any]:
    valid = np.where(~np.isnan(grid_scores))[0]
    if valid.size:
        best_unit = int(valid[np.argsort(grid_scores[valid])[::-1][0]])
        best_grid_score = float(grid_scores[best_unit])
        mean_grid_score = float(np.mean(grid_scores[valid]))
    else:
        best_unit = -1
        best_grid_score = float("nan")
        mean_grid_score = float("nan")
    return {
        "best_grid_score": best_grid_score,
        "best_unit": best_unit,
        "mean_grid_score": mean_grid_score,
        "valid_units": int(valid.size),
        "unit_count": int(grid_scores.shape[0]),
        "num_steps": int(positions.shape[0]),
        "bounds": [float(value) for value in bounds],
        "n_bins": int(n_bins),
        "max_steps": None if max_steps is None else int(max_steps),
        "max_units": None if max_units is None else int(max_units),
        "top_k": int(top_k),
    }


def plot_top_grid_cells(
    ratemaps: np.ndarray,
    autocorrs: np.ndarray,
    grid_scores: np.ndarray,
    output_dir: Path,
    top_k: int,
) -> None:
    valid = np.where(~np.isnan(grid_scores))[0]
    if valid.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No valid grid scores", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "top_grid_cells.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    top = valid[np.argsort(grid_scores[valid])[::-1]][:top_k]
    fig, axes = plt.subplots(len(top), 2, figsize=(8, 3.2 * len(top)), squeeze=False)
    for row, unit in enumerate(top):
        im0 = axes[row, 0].imshow(ratemaps[unit].T, origin="lower", cmap="jet", interpolation="nearest")
        axes[row, 0].set_title(f"Unit {unit} ratemap")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        im1 = axes[row, 1].imshow(autocorrs[unit].T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
        axes[row, 1].set_title(f"Grid score {grid_scores[unit]:.3f}")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "top_grid_cells.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ratemap_grid(ratemaps: np.ndarray, output_dir: Path, max_units: int = 64) -> None:
    count = min(max_units, ratemaps.shape[0])
    if count == 0:
        return
    n_cols = min(8, count)
    n_rows = int(np.ceil(count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= count:
            ax.axis("off")
            continue
        ax.imshow(ratemaps[idx].T, origin="lower", cmap="jet", interpolation="nearest")
        ax.set_title(f"Unit {idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_ratemaps_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_gridscore_analysis(
    model: Any,
    dataset_root: str | Path,
    *,
    batch_size_sequences: int,
    max_seq_len: int | None,
    n_bins: int,
    bounds: tuple[float, float, float, float] | None = None,
    max_steps: int | None = None,
    max_units: int | None = 256,
    top_k: int = 8,
    gridscore_positive_activations: bool = False,
) -> dict[str, Any]:
    positions, activations = collect_bottleneck_activity(
        model,
        dataset_root,
        batch_size_sequences=batch_size_sequences,
        max_seq_len=max_seq_len,
        max_steps=max_steps,
        max_units=max_units,
    )
    gridscore_activations = np.maximum(activations, 0.0) if gridscore_positive_activations else activations
    resolved_bounds = resolve_gridscore_bounds(positions, bounds)
    ratemaps = compute_spatial_ratemaps(positions, gridscore_activations, n_bins=n_bins, bounds=resolved_bounds)
    autocorrs, grid_scores = analyze_grid_scores(ratemaps)
    summary = summarize_grid_scores(
        positions,
        grid_scores,
        bounds=resolved_bounds,
        n_bins=n_bins,
        max_steps=max_steps,
        max_units=max_units,
        top_k=top_k,
    )
    summary["gridscore_positive_activations"] = bool(gridscore_positive_activations)
    return {
        "positions": positions,
        "activations": activations,
        "ratemaps": ratemaps,
        "autocorrs": autocorrs,
        "grid_scores": grid_scores,
        "bounds": resolved_bounds,
        "summary": summary,
    }


def save_gridscore_analysis(
    analysis: dict[str, Any],
    output_dir: Path,
    *,
    data_filename: str,
    summary_filename: str | None,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / data_filename,
        positions=analysis["positions"],
        activations=analysis["activations"],
        ratemaps=analysis["ratemaps"],
        autocorrs=analysis["autocorrs"],
        grid_scores=analysis["grid_scores"],
        bounds=np.asarray(analysis["bounds"], dtype=np.float32),
    )
    if summary_filename is not None:
        (output_dir / summary_filename).write_text(
            json.dumps(analysis["summary"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    plot_top_grid_cells(
        analysis["ratemaps"],
        analysis["autocorrs"],
        analysis["grid_scores"],
        output_dir,
        top_k,
    )
    plot_ratemap_grid(analysis["ratemaps"], output_dir)


def export_gridscore_artifacts(
    model: Any,
    dataset_root: str | Path,
    output_dir: Path,
    *,
    batch_size_sequences: int,
    max_seq_len: int | None,
    n_bins: int,
    bounds: tuple[float, float, float, float] | None = None,
    max_steps: int | None = None,
    max_units: int | None = 256,
    top_k: int = 8,
    gridscore_positive_activations: bool = False,
) -> dict[str, Any]:
    analysis = compute_gridscore_analysis(
        model,
        dataset_root,
        batch_size_sequences=batch_size_sequences,
        max_seq_len=max_seq_len,
        n_bins=n_bins,
        bounds=bounds,
        max_steps=max_steps,
        max_units=max_units,
        top_k=top_k,
        gridscore_positive_activations=gridscore_positive_activations,
    )
    save_gridscore_analysis(
        analysis,
        output_dir,
        data_filename="gridscore_data.npz",
        summary_filename="gridscore_summary.json",
        top_k=top_k,
    )
    return analysis["summary"]


def gridscore_metrics_from_summary(summary: dict[str, Any], *, seconds: float) -> dict[str, float]:
    return {
        "offline_pi/probe/gridscore/best": float(summary["best_grid_score"]),
        "offline_pi/probe/gridscore/best_unit": float(summary["best_unit"]),
        "offline_pi/probe/gridscore/mean": float(summary["mean_grid_score"]),
        "offline_pi/probe/gridscore/valid_units": float(summary["valid_units"]),
        "offline_pi/probe/gridscore/seconds": float(seconds),
    }
