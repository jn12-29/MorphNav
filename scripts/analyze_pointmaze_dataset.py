from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components.dataset_gen.pointmaze_config import (  # noqa: E402
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PointMaze dataset coverage and action distribution.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"))
    parser.add_argument("--max-preview-episodes", type=int, default=12)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _validate_shard_schema_values(schema: Any, version: Any, shard_path: Path) -> None:
    if schema != POINTMAZE_MUJOCO_ZARR_SCHEMA or int(version or -1) != POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION:
        raise ValueError(
            f"{shard_path} has unsupported dataset schema {schema!r} version {version!r}; "
            f"expected {POINTMAZE_MUJOCO_ZARR_SCHEMA!r} version {POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION}"
        )


def _validate_shard_schema(root: zarr.Group, shard_path: Path) -> None:
    _validate_shard_schema_values(root.attrs.get("dataset_schema"), root.attrs.get("dataset_schema_version"), shard_path)


def _shard_paths(dataset_root: Path) -> list[Path]:
    shard_paths = sorted(dataset_root.glob("shard_*.npz")) + sorted(dataset_root.glob("shard_*.zarr"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz or shard_*.zarr files found under {dataset_root}")
    return shard_paths


def _load_zarr_distribution_shard(shard_path: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = zarr.open_group(str(shard_path), mode="r")
    _validate_shard_schema(root, shard_path)
    if not metadata and "dataset_meta_json" in root.attrs:
        metadata = json.loads(root.attrs["dataset_meta_json"])
    return metadata, {
        "lengths": np.asarray(root["episode_lengths"][:], dtype=np.int64),
        "offsets": np.asarray(root["episode_offsets"][:], dtype=np.int64),
        "actions": np.asarray(root["step/action"][:], dtype=np.float32),
        "positions": np.asarray(root["step/qpos"][:], dtype=np.float32)[..., :2],
        "obs_positions": np.asarray(root["obs/achieved_goal"][:], dtype=np.float32)[..., :2],
        "rewards": np.asarray(root["step/reward"][:], dtype=np.float32),
        "terminated": np.asarray(root["step/terminated"][:], dtype=bool),
        "truncated": np.asarray(root["step/truncated"][:], dtype=bool),
    }


def _load_npz_distribution_shard(shard_path: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(shard_path, allow_pickle=False) as data:
        shard_metadata = json.loads(str(data["dataset_meta_json"]))
        _validate_shard_schema_values(shard_metadata.get("dataset_schema"), shard_metadata.get("dataset_schema_version"), shard_path)
        if not metadata:
            metadata = shard_metadata
        shard = {
            "lengths": np.asarray(data["episode_lengths"], dtype=np.int64),
            "offsets": np.asarray(data["episode_offsets"], dtype=np.int64),
            "actions": np.asarray(data["step/action"], dtype=np.float32),
            "positions": np.asarray(data["step/qpos"], dtype=np.float32)[..., :2],
            "obs_positions": np.asarray(data["obs/achieved_goal"], dtype=np.float32)[..., :2],
            "rewards": np.asarray(data["step/reward"], dtype=np.float32),
            "terminated": np.asarray(data["step/terminated"], dtype=bool),
            "truncated": np.asarray(data["step/truncated"], dtype=bool),
        }
    return metadata, shard


def _load_distribution_shard(shard_path: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if shard_path.suffix == ".npz":
        return _load_npz_distribution_shard(shard_path, metadata)
    return _load_zarr_distribution_shard(shard_path, metadata)


def load_dataset_distribution(dataset_root: str | Path, max_preview_episodes: int) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    lengths_parts = []
    actions_parts = []
    positions_parts = []
    obs_positions_parts = []
    rewards_parts = []
    terminal_terminated = []
    terminal_truncated = []
    preview_episodes = []

    for shard_path in _shard_paths(dataset_root):
        metadata, shard = _load_distribution_shard(shard_path, metadata)
        lengths = shard["lengths"]
        offsets = shard["offsets"]
        actions = shard["actions"]
        positions = shard["positions"]
        obs_positions = shard["obs_positions"]
        rewards = shard["rewards"]
        terminated = shard["terminated"]
        truncated = shard["truncated"]

        lengths_parts.append(lengths)
        actions_parts.append(actions)
        positions_parts.append(positions)
        obs_positions_parts.append(obs_positions)
        rewards_parts.append(rewards)
        for offset, length in zip(offsets, lengths, strict=True):
            end = int(offset + length)
            terminal_terminated.append(bool(terminated[end - 1]))
            terminal_truncated.append(bool(truncated[end - 1]))
            if len(preview_episodes) < max_preview_episodes:
                preview_episodes.append(positions[int(offset) : end])

    return {
        "metadata": metadata,
        "lengths": np.concatenate(lengths_parts),
        "actions": np.concatenate(actions_parts, axis=0),
        "positions": np.concatenate(positions_parts, axis=0),
        "obs_positions": np.concatenate(obs_positions_parts, axis=0),
        "rewards": np.concatenate(rewards_parts, axis=0),
        "terminal_terminated": np.asarray(terminal_terminated, dtype=bool),
        "terminal_truncated": np.asarray(terminal_truncated, dtype=bool),
        "preview_episodes": preview_episodes,
    }


def summarize_distribution(data: dict[str, Any], *, n_bins: int, bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    lengths = np.asarray(data["lengths"])
    actions = np.asarray(data["actions"])
    positions = np.asarray(data["positions"])
    obs_positions = np.asarray(data["obs_positions"])
    rewards = np.asarray(data["rewards"])
    hist, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=n_bins,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
    )
    occupancy = hist.astype(np.float64)
    total_occupancy = float(occupancy.sum())
    nonempty_bins = int(np.count_nonzero(occupancy))
    if total_occupancy > 0.0:
        prob = occupancy.ravel() / total_occupancy
        prob = prob[prob > 0.0]
        occupancy_entropy = float(-(prob * np.log(prob)).sum() / np.log(n_bins * n_bins))
    else:
        occupancy_entropy = 0.0
    nonzero_occupancy = occupancy[occupancy > 0.0]
    mean_bin_count = float(occupancy.mean())
    occupancy_cv = float(occupancy.std() / mean_bin_count) if mean_bin_count > 0.0 else 0.0
    occupancy_max_to_mean = float(occupancy.max() / mean_bin_count) if mean_bin_count > 0.0 else 0.0
    nonzero_min_to_mean = float(nonzero_occupancy.min() / mean_bin_count) if mean_bin_count > 0.0 and nonzero_occupancy.size else 0.0
    half_width_x = 0.25 * (bounds[1] - bounds[0])
    half_width_y = 0.25 * (bounds[3] - bounds[2])
    center_x = 0.5 * (bounds[0] + bounds[1])
    center_y = 0.5 * (bounds[2] + bounds[3])
    center_mask = (
        (np.abs(positions[:, 0] - center_x) <= half_width_x)
        & (np.abs(positions[:, 1] - center_y) <= half_width_y)
    )
    uniform_std = np.asarray(
        [
            (bounds[1] - bounds[0]) / np.sqrt(12.0),
            (bounds[3] - bounds[2]) / np.sqrt(12.0),
        ],
        dtype=np.float64,
    )

    step_delta = positions - obs_positions
    metadata = data["metadata"]
    return {
        "dataset_name": metadata.get("dataset_name"),
        "dataset_seed": metadata.get("dataset_seed"),
        "policy_type": metadata.get("policy_type"),
        "policy_params": metadata.get("policy_params"),
        "env_kwargs": metadata.get("env_kwargs"),
        "num_episodes": int(lengths.shape[0]),
        "num_steps": int(lengths.sum()),
        "episode_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        },
        "terminal_rate": {
            "terminated": float(np.asarray(data["terminal_terminated"]).mean()),
            "truncated": float(np.asarray(data["terminal_truncated"]).mean()),
        },
        "reward": {
            "sum": float(rewards.sum()),
            "mean": float(rewards.mean()),
        },
        "position": {
            "min": positions.min(axis=0),
            "max": positions.max(axis=0),
            "mean": positions.mean(axis=0),
            "std": positions.std(axis=0),
        },
        "action": {
            "min": actions.min(axis=0),
            "max": actions.max(axis=0),
            "mean": actions.mean(axis=0),
            "std": actions.std(axis=0),
        },
        "step_displacement": {
            "mean_norm": float(np.linalg.norm(step_delta, axis=1).mean()),
            "std_norm": float(np.linalg.norm(step_delta, axis=1).std()),
        },
        "occupancy": {
            "bounds": bounds,
            "n_bins": int(n_bins),
            "nonempty_bins": nonempty_bins,
            "coverage_fraction": float(nonempty_bins / float(n_bins * n_bins)),
            "normalized_entropy": occupancy_entropy,
            "coefficient_of_variation": occupancy_cv,
            "max_to_mean": occupancy_max_to_mean,
            "nonzero_min_to_mean": nonzero_min_to_mean,
            "central_half_fraction": float(center_mask.mean()),
            "position_std_fraction_of_uniform": positions.std(axis=0) / uniform_std,
        },
    }


def resolve_bounds(data: dict[str, Any], raw_bounds: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float]:
    if raw_bounds is not None:
        bounds = tuple(float(v) for v in raw_bounds)
        if len(bounds) != 4:
            raise ValueError("bounds must contain four values")
        return bounds  # type: ignore[return-value]

    policy_params = data.get("metadata", {}).get("policy_params", {})
    if isinstance(policy_params, dict) and "arena_min" in policy_params and "arena_max" in policy_params:
        arena_min = float(policy_params["arena_min"])
        arena_max = float(policy_params["arena_max"])
        return (arena_min, arena_max, arena_min, arena_max)
    return (-2.5, 2.5, -2.5, 2.5)


def plot_distribution(data: dict[str, Any], output_dir: Path, *, n_bins: int, bounds: tuple[float, float, float, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(data["positions"])
    actions = np.asarray(data["actions"])

    fig, ax = plt.subplots(figsize=(6, 5))
    hist, x_edges, y_edges = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=n_bins,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
    )
    image = ax.imshow(
        hist.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("PointMaze occupancy")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, label="steps")
    fig.tight_layout()
    fig.savefig(output_dir / "occupancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(actions[:, 0], bins=60, range=(-1.0, 1.0), color="#4c72b0")
    axes[0].set_title("action x")
    axes[1].hist(actions[:, 1], bins=60, range=(-1.0, 1.0), color="#dd8452")
    axes[1].set_title("action y")
    for ax in axes:
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "action_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for episode in data["preview_episodes"]:
        if episode.shape[0] > 0:
            ax.plot(episode[:, 0], episode[:, 1], linewidth=0.8, alpha=0.8)
            ax.scatter(episode[0, 0], episode[0, 1], s=8, color="green", alpha=0.8)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Trajectory preview")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_bins <= 0:
        parser.error("--n-bins must be > 0")
    if args.max_preview_episodes < 0:
        parser.error("--max-preview-episodes must be >= 0")
    data = load_dataset_distribution(args.dataset_root, max_preview_episodes=args.max_preview_episodes)
    bounds = resolve_bounds(data, args.bounds)
    if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        parser.error("--bounds must satisfy MIN_X < MAX_X and MIN_Y < MAX_Y")

    summary = summarize_distribution(data, n_bins=args.n_bins, bounds=bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_distribution.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_distribution(data, args.output_dir, n_bins=args.n_bins, bounds=bounds)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
