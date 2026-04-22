from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MANIFEST_SCHEMA = "pointmaze_episode_plan"
MANIFEST_VERSION = 1


@dataclass
class EpisodePlanItem:
    episode_id: int
    shard_id: int
    episode_seed: int


@dataclass
class EpisodePlan:
    dataset_seed: int
    episodes: list[EpisodePlanItem]


def build_episode_plan(
    dataset_seed: int,
    num_episodes: int,
    episodes_per_shard: int,
) -> EpisodePlan:
    if num_episodes < 0:
        raise ValueError("num_episodes must be >= 0")
    if episodes_per_shard <= 0:
        raise ValueError("episodes_per_shard must be > 0")

    seed_sequence = np.random.SeedSequence(dataset_seed)
    child_sequences = seed_sequence.spawn(num_episodes)

    episodes: list[EpisodePlanItem] = []
    for episode_id, child in enumerate(child_sequences):
        episode_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        shard_id = episode_id // episodes_per_shard
        episodes.append(
            EpisodePlanItem(
                episode_id=episode_id,
                shard_id=shard_id,
                episode_seed=episode_seed,
            )
        )

    return EpisodePlan(dataset_seed=dataset_seed, episodes=episodes)


def save_manifest(plan: EpisodePlan, path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "dataset_seed": plan.dataset_seed,
        "episodes": [
            {
                "episode_id": item.episode_id,
                "shard_id": item.shard_id,
                "episode_seed": item.episode_seed,
            }
            for item in plan.episodes
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> EpisodePlan:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest payload must be a JSON object")

    schema = payload.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(f"Invalid manifest schema: expected {MANIFEST_SCHEMA!r}, got {schema!r}")

    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"Invalid manifest version: expected {MANIFEST_VERSION!r}, got {version!r}")

    dataset_seed = payload.get("dataset_seed")
    if not isinstance(dataset_seed, int):
        raise ValueError("Manifest field 'dataset_seed' must be an integer")

    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list):
        raise ValueError("Manifest field 'episodes' must be a list")

    episodes: list[EpisodePlanItem] = []
    for idx, item in enumerate(episodes_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Episode item at index {idx} must be an object")
        episode_id = item.get("episode_id")
        shard_id = item.get("shard_id")
        episode_seed = item.get("episode_seed")
        if not isinstance(episode_id, int):
            raise ValueError(f"Episode item {idx} field 'episode_id' must be an integer")
        if not isinstance(shard_id, int):
            raise ValueError(f"Episode item {idx} field 'shard_id' must be an integer")
        if not isinstance(episode_seed, int):
            raise ValueError(f"Episode item {idx} field 'episode_seed' must be an integer")
        episodes.append(
            EpisodePlanItem(
                episode_id=episode_id,
                shard_id=shard_id,
                episode_seed=episode_seed,
            )
        )

    return EpisodePlan(dataset_seed=dataset_seed, episodes=episodes)
