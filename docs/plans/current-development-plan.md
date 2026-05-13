# Current Development Plan

## Documentation Policy

- `README.md` is the current user entry point.
- `CLAUDE.md` is a short mistake-prevention note for Claude Code.
- `AGENTS.md` is intentionally not maintained.
- The old `docs/superpowers/` plugin documents are no longer the source of truth.

## Completed

- PointMaze dataset generator first version.
- Standalone CLI: `scripts/generate_pointmaze_dataset.py`.
- Dataset modules under `components/dataset_gen/`.
- Tests for manifest planning, weak random policy, spatial annotations, collector behavior, Zarr writing, and CLI smoke coverage.

## Current Dataset Contract

The current PointMaze dataset contract follows the implementation in `components/dataset_gen/` and `scripts/generate_pointmaze_dataset.py`.

- Output root: `<output-dir>/<dataset-name>/`.
- Metadata files: `manifest.json` and `dataset_metadata.json`.
- Shards: `shard_000000.zarr`, `shard_000001.zarr`, and so on.
- Step arrays: `step/action`, `step/reward`, `step/terminated`, `step/truncated`, `step/qpos`, `step/qvel`, `step/goal`.
- Annotation arrays: `annotation/agent_xy`, `annotation/heading`, `annotation/goal_xy`, `annotation/relative_goal`.
- Episode index arrays: `episode_lengths`, `episode_offsets`.
- Shard attributes include `dataset_meta_json`, `episode_summaries_json`, and JSON-safe scalar dataset metadata keys.

The first version does not persist full `obs` or raw `info` payloads into Zarr shards.

## In Progress

- `pi_ppo_lstm` path-integration auxiliary training.
- Core files: `components/path_integration.py`, `components/pi_algo.py`, `components/pi_policy.py`.
- Training-stack integration: `rl-baselines3-zoo/conf/maze_pi.yml`, `rl-baselines3-zoo/rl_zoo3/utils.py`, and `rl-baselines3-zoo/rl_zoo3/exp_manager.py`.
- Documentation entry points: `README.md`, `CLAUDE.md`, and `scripts/sb3zoo_train.sh`.

## Next Steps

- Add minimal PI import/config tests or a smoke check.
- Verify `pi_ppo_lstm` registration through the local zoo entry point.
- Decide later whether dataset shards should persist full observations and selected `info` fields.
