import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "components" / "dataset_gen" / "pointmaze_manifest.py"
SPEC = importlib.util.spec_from_file_location("pointmaze_manifest", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load pointmaze_manifest module at {MODULE_PATH}")
pointmaze_manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pointmaze_manifest
SPEC.loader.exec_module(pointmaze_manifest)

EpisodePlan = pointmaze_manifest.EpisodePlan
EpisodePlanItem = pointmaze_manifest.EpisodePlanItem
build_episode_plan = pointmaze_manifest.build_episode_plan
load_manifest = pointmaze_manifest.load_manifest
save_manifest = pointmaze_manifest.save_manifest


def _expected_episode_seeds(dataset_seed: int, num_episodes: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(dataset_seed)
    children = seed_sequence.spawn(num_episodes)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def test_build_episode_plan_is_deterministic_and_worker_independent():
    dataset_seed = 42
    num_episodes = 11
    episodes_per_shard = 4

    first_plan = build_episode_plan(
        dataset_seed=dataset_seed,
        num_episodes=num_episodes,
        episodes_per_shard=episodes_per_shard,
    )
    second_plan = build_episode_plan(
        dataset_seed=dataset_seed,
        num_episodes=num_episodes,
        episodes_per_shard=episodes_per_shard,
    )

    assert first_plan == second_plan

    expected_seeds = _expected_episode_seeds(dataset_seed, num_episodes)
    actual_seeds = [item.episode_seed for item in first_plan.episodes]
    assert actual_seeds == expected_seeds

    for item in first_plan.episodes:
        assert item.shard_id == item.episode_id // episodes_per_shard


def test_save_and_load_manifest_roundtrip(tmp_path: Path):
    plan = EpisodePlan(
        dataset_seed=9,
        episodes=[
            EpisodePlanItem(episode_id=0, shard_id=0, episode_seed=123),
            EpisodePlanItem(episode_id=1, shard_id=0, episode_seed=456),
            EpisodePlanItem(episode_id=2, shard_id=1, episode_seed=789),
        ],
    )

    manifest_path = tmp_path / "nested" / "manifest.json"
    save_manifest(plan, manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pointmaze_episode_plan"
    assert payload["version"] == 1

    loaded = load_manifest(manifest_path)
    assert loaded == plan


def test_load_manifest_invalid_payload_raises_value_error(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "pointmaze_episode_plan",
                "version": 1,
                "dataset_seed": 0,
                "episodes": [{"episode_id": 0, "shard_id": 0}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="episode_seed"):
        load_manifest(manifest_path)
