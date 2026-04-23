import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

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

SCRIPT_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_pointmaze_dataset.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("generate_pointmaze_dataset", SCRIPT_MODULE_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Cannot load generate_pointmaze_dataset module at {SCRIPT_MODULE_PATH}")
generate_pointmaze_dataset = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = generate_pointmaze_dataset
SCRIPT_SPEC.loader.exec_module(generate_pointmaze_dataset)


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


def test_generate_pointmaze_dataset_cli_help_surface():
    script_path = SCRIPT_MODULE_PATH
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    help_text = result.stdout
    for flag in (
        "--output-dir",
        "--dataset-name",
        "--num-episodes",
        "--episodes-per-shard",
        "--dataset-seed",
        "--maze-map-name",
        "--max-episode-steps",
    ):
        assert flag in help_text


def test_generate_dataset_removes_stale_shards_before_writing(monkeypatch, tmp_path: Path):
    stale_shard_dir = tmp_path / "dataset" / "shard_000123.zarr"
    stale_shard_dir.mkdir(parents=True)
    stale_shard_file = tmp_path / "dataset" / "shard_000456.zarr"
    stale_shard_file.write_text("stale", encoding="utf-8")
    untouched = tmp_path / "dataset" / "shard_notes.zarr"
    untouched.write_text("keep", encoding="utf-8")

    class _DummyEnv:
        def close(self):
            return None

    class _DummyPolicy:
        def __init__(self, config, seed):
            self.config = config
            self.seed = seed

    @dataclass
    class _PolicyConfig:
        turn_bias: float = 0.0

    monkeypatch.setattr(
        "components.dataset_gen.pointmaze_manifest.build_episode_plan",
        lambda dataset_seed, num_episodes, episodes_per_shard: EpisodePlan(
            dataset_seed=dataset_seed,
            episodes=[EpisodePlanItem(episode_id=0, shard_id=0, episode_seed=111)],
        ),
    )
    monkeypatch.setattr(
        "components.dataset_gen.pointmaze_manifest.save_manifest",
        lambda plan, path: path.write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr("components.dataset_gen.pointmaze_env_factory.build_pointmaze_env_kwargs", lambda config: {})
    monkeypatch.setattr("components.dataset_gen.pointmaze_env_factory.create_pointmaze_env", lambda config: _DummyEnv())
    monkeypatch.setattr("components.dataset_gen.pointmaze_policy.WeakRandomPolicyDriver", _DummyPolicy)
    monkeypatch.setattr(
        "components.dataset_gen.pointmaze_collector.collect_episode",
        lambda **kwargs: {
            "episode_id": kwargs["episode_id"],
            "episode_seed": kwargs["episode_seed"],
            "reward": np.zeros((1,), dtype=np.float32),
            "action": np.zeros((1, 2), dtype=np.float32),
            "terminated": np.zeros((1,), dtype=bool),
            "truncated": np.zeros((1,), dtype=bool),
            "qpos": np.zeros((1, 2), dtype=np.float32),
            "qvel": np.zeros((1, 2), dtype=np.float32),
            "goal": np.zeros((1, 2), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "components.dataset_gen.pointmaze_annotation.annotate_episode",
        lambda episode: {
            "agent_xy": np.zeros((1, 2), dtype=np.float32),
            "heading": np.zeros((1,), dtype=np.float32),
            "goal_xy": np.zeros((1, 2), dtype=np.float32),
            "relative_goal": np.zeros((1, 2), dtype=np.float32),
        },
    )
    fake_zarr_writer = types.ModuleType("components.dataset_gen.pointmaze_zarr_writer")
    fake_zarr_writer.write_shard = lambda output_dir, shard_id, episodes, dataset_meta: (
        output_dir / f"shard_{shard_id:06d}.zarr"
    ).mkdir()
    fake_zarr_writer.write_dataset_metadata = lambda output_dir, dataset_meta: (output_dir / "dataset_metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "components.dataset_gen.pointmaze_zarr_writer", fake_zarr_writer)

    config = SimpleNamespace(
        output=SimpleNamespace(output_dir=str(tmp_path), dataset_name="dataset", episodes_per_shard=1),
        env=SimpleNamespace(env_id="PointMaze", maze_map_name="OPEN", max_episode_steps=1000),
        policy=_PolicyConfig(),
        dataset_seed=0,
        num_episodes=1,
    )

    generate_pointmaze_dataset.generate_dataset(config)

    assert not stale_shard_dir.exists()
    assert not stale_shard_file.exists()
    assert untouched.exists()


def test_generate_pointmaze_dataset_cli_smoke(tmp_path: Path):
    script_path = SCRIPT_MODULE_PATH
    output_dir = tmp_path / "output"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    vendor_dir = Path(__file__).resolve().parents[1] / ".vendor"
    if vendor_dir.exists():
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{vendor_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(vendor_dir)
        )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            "2",
            "--episodes-per-shard",
            "2",
            "--dataset-seed",
            "5",
            "--max-episode-steps",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr

    dataset_dir = output_dir / "pointmaze_mujoco"
    assert (dataset_dir / "dataset_metadata.json").exists()
    assert (dataset_dir / "manifest.json").exists()
    assert any(dataset_dir.glob("shard_*.zarr"))
