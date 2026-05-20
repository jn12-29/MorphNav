from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from scipy.ndimage import rotate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import PathIntegrationRecurrentPPO
from components.dataset_gen.pointmaze_config import make_phase1_pointmaze_pi_env_config
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs
from components.offline_pi_rehearsal import load_offline_pi_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze offline PI bottleneck spatial representations.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size-sequences", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def resolve_output_dir(model_path: Path, dataset_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    run_root = model_path.parent.parent if model_path.parent.name == "models" else model_path.parent
    return run_root / "analysis" / "representations" / dataset_root.name


def _make_env():
    env_config = make_phase1_pointmaze_pi_env_config()
    return gym.make("PointMaze", **build_pointmaze_env_kwargs(env_config))


def collect_bottleneck_activity(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    env = _make_env()
    try:
        model = PathIntegrationRecurrentPPO.load(args.model_path, env=env, device=args.device)
        policy = model.policy
        policy.set_training_mode(False)
        n_lstm_layers, _n_envs, lstm_hidden_size = policy.lstm_hidden_state_shape

        positions = []
        activations = []
        with th.no_grad():
            for batch in load_offline_pi_batches(
                args.dataset_root,
                batch_size_sequences=args.batch_size_sequences,
                max_seq_len=args.max_seq_len,
                shuffle=False,
                device=policy.device,
                n_lstm_layers=int(n_lstm_layers),
                lstm_hidden_size=int(lstm_hidden_size),
            ):
                outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
                mask = batch.mask.bool()
                positions.append(batch.target_pos[mask].detach().cpu().numpy())
                activations.append(outputs.bottleneck[mask].detach().cpu().numpy())
                if args.max_steps is not None and sum(part.shape[0] for part in positions) >= args.max_steps:
                    break

        if not positions:
            raise ValueError("No activity was collected from the dataset")
        position_arr = np.concatenate(positions, axis=0)
        activation_arr = np.concatenate(activations, axis=0)
        if args.max_steps is not None:
            position_arr = position_arr[: args.max_steps]
            activation_arr = activation_arr[: args.max_steps]
        if args.max_units is not None:
            activation_arr = activation_arr[:, : args.max_units]
        return position_arr, activation_arr
    finally:
        env.close()


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


def plot_top_grid_cells(ratemaps: np.ndarray, autocorrs: np.ndarray, grid_scores: np.ndarray, output_dir: Path, top_k: int) -> None:
    valid = np.where(~np.isnan(grid_scores))[0]
    if valid.size == 0:
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size_sequences <= 0:
        parser.error("--batch-size-sequences must be > 0")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be > 0")

    output_dir = resolve_output_dir(args.model_path, args.dataset_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, activations = collect_bottleneck_activity(args)
    if args.bounds is None:
        margin = 0.1
        bounds = (
            float(np.min(positions[:, 0]) - margin),
            float(np.max(positions[:, 0]) + margin),
            float(np.min(positions[:, 1]) - margin),
            float(np.max(positions[:, 1]) + margin),
        )
    else:
        bounds = tuple(float(v) for v in args.bounds)

    ratemaps = compute_spatial_ratemaps(positions, activations, n_bins=args.n_bins, bounds=bounds)
    autocorrs, grid_scores = analyze_grid_scores(ratemaps)
    np.savez(
        output_dir / "analysis_data.npz",
        positions=positions,
        activations=activations,
        ratemaps=ratemaps,
        autocorrs=autocorrs,
        grid_scores=grid_scores,
        bounds=np.asarray(bounds, dtype=np.float32),
    )
    plot_top_grid_cells(ratemaps, autocorrs, grid_scores, output_dir, args.top_k)
    plot_ratemap_grid(ratemaps, output_dir)

    valid = np.where(~np.isnan(grid_scores))[0]
    if valid.size:
        best = valid[np.argsort(grid_scores[valid])[::-1][0]]
        print(f"Best grid score: {grid_scores[best]:.6f} (unit {best})")
    else:
        print("No valid grid scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
