from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from components.dataset_gen.pointmaze_config import PointMazeDatasetConfig
    from components.dataset_gen.pointmaze_manifest import EpisodePlanItem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PointMaze MuJoCo dataset shards.")
    parser.add_argument("--output-dir", type=Path, default=Path("recorded_data/pointmaze"))
    parser.add_argument("--dataset-name", type=str, default="pointmaze_mujoco")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-shard", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--maze-map-name", type=str, default="OPEN")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    return parser


def _build_config(args: argparse.Namespace) -> "PointMazeDatasetConfig":
    from components.dataset_gen.pointmaze_config import (
        PointMazeDatasetConfig,
        PointMazeEnvConfig,
        PointMazeOutputConfig,
    )

    env_config = PointMazeEnvConfig(
        maze_map_name=args.maze_map_name,
        max_episode_steps=args.max_episode_steps,
    )
    output_config = PointMazeOutputConfig(
        output_dir=str(args.output_dir),
        dataset_name=args.dataset_name,
        episodes_per_shard=args.episodes_per_shard,
        num_workers=1,
    )
    return PointMazeDatasetConfig(
        env=env_config,
        output=output_config,
        dataset_seed=args.dataset_seed,
        num_episodes=args.num_episodes,
    )


def _group_by_shard(items: Iterable["EpisodePlanItem"]) -> dict[int, list["EpisodePlanItem"]]:
    grouped: dict[int, list["EpisodePlanItem"]] = {}
    for item in items:
        grouped.setdefault(item.shard_id, []).append(item)
    return grouped


def _cleanup_existing_shards(dataset_dir: Path) -> None:
    pattern = "shard_[0-9][0-9][0-9][0-9][0-9][0-9].zarr"
    for shard_path in dataset_dir.glob(pattern):
        if shard_path.is_dir():
            shutil.rmtree(shard_path)
        else:
            shard_path.unlink(missing_ok=True)


def generate_dataset(config: "PointMazeDatasetConfig") -> Path:
    from dataclasses import asdict

    from components.dataset_gen.pointmaze_annotation import annotate_episode
    from components.dataset_gen.pointmaze_collector import collect_episode
    from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs, create_pointmaze_env
    from components.dataset_gen.pointmaze_manifest import build_episode_plan, save_manifest
    from components.dataset_gen.pointmaze_policy import WeakRandomPolicyDriver
    from components.dataset_gen.pointmaze_zarr_writer import write_dataset_metadata, write_shard

    dataset_dir = Path(config.output.output_dir) / config.output.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_existing_shards(dataset_dir)

    episode_plan = build_episode_plan(
        dataset_seed=config.dataset_seed,
        num_episodes=config.num_episodes,
        episodes_per_shard=config.output.episodes_per_shard,
    )
    save_manifest(episode_plan, dataset_dir / "manifest.json")

    shard_to_items = _group_by_shard(episode_plan.episodes)
    env_kwargs = build_pointmaze_env_kwargs(config.env)
    dataset_meta = {
        "dataset_name": config.output.dataset_name,
        "dataset_seed": config.dataset_seed,
        "num_episodes": config.num_episodes,
        "episodes_per_shard": config.output.episodes_per_shard,
        "num_shards": len(shard_to_items),
        "env_id": config.env.env_id,
        "maze_map_name": config.env.maze_map_name,
        "max_episode_steps": config.env.max_episode_steps,
        "env_kwargs": env_kwargs,
        "policy_type": "WeakRandomPolicyDriver",
        "policy_params": asdict(config.policy),
    }

    for shard_id in sorted(shard_to_items):
        episodes: list[dict] = []
        for item in shard_to_items[shard_id]:
            env = create_pointmaze_env(config.env)
            try:
                policy = WeakRandomPolicyDriver(config=config.policy, seed=item.episode_seed)
                env_metadata = {
                    "env_id": config.env.env_id,
                    "maze_map_name": config.env.maze_map_name,
                    "max_episode_steps": config.env.max_episode_steps,
                    "env_kwargs": env_kwargs,
                }
                policy_metadata = {
                    "policy_type": "WeakRandomPolicyDriver",
                    "policy_params": asdict(config.policy),
                }
                episode = collect_episode(
                    env=env,
                    policy=policy,
                    episode_id=item.episode_id,
                    episode_seed=item.episode_seed,
                    env_metadata=env_metadata,
                    policy_metadata=policy_metadata,
                )
            finally:
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()

            episode["annotation"] = annotate_episode(episode)
            episodes.append(episode)

        write_shard(
            output_dir=dataset_dir,
            shard_id=shard_id,
            episodes=episodes,
            dataset_meta=dataset_meta,
        )

    write_dataset_metadata(dataset_dir, dataset_meta)
    return dataset_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.num_episodes < 0:
        parser.error("--num-episodes must be >= 0")
    if args.episodes_per_shard <= 0:
        parser.error("--episodes-per-shard must be > 0")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be > 0")

    config = _build_config(args)
    dataset_dir = generate_dataset(config)
    print(f"Wrote dataset to {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
