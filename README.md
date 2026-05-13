# MorphNav

## Install

Follow `scripts/build_conda_env.sh`.

## PointMaze Dataset Generation

The standalone generator writes the current dataset format implemented by `components/dataset_gen/`.

```bash
python scripts/generate_pointmaze_dataset.py \
  --output-dir recorded_data \
  --dataset-name pointmaze_mujoco \
  --num-episodes 1000 \
  --episodes-per-shard 100 \
  --dataset-seed 0 \
  --maze-map-name OPEN \
  --max-episode-steps 1000
```

This writes to `<output-dir>/<dataset-name>/`:

- `manifest.json`
- `dataset_metadata.json`
- `shard_000000.zarr`, `shard_000001.zarr`, and so on

Shard arrays include `episode_lengths`, `episode_offsets`, `step/action`, `step/reward`, `step/terminated`, `step/truncated`, `step/qpos`, `step/qvel`, `step/goal`, `annotation/agent_xy`, `annotation/heading`, `annotation/goal_xy`, and `annotation/relative_goal`.

Shard attributes include `dataset_meta_json`, `episode_summaries_json`, and JSON-safe scalar dataset metadata keys.

The current first version does not persist full `obs` or raw `info` payloads into Zarr shards.

## Train

Follow `scripts/sb3zoo_train.sh` and inspect the command before running it.

```bash
python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml
```

### Path integration auxiliary training

`pi_ppo_lstm` adds a parallel place-cell prediction branch to PPO-LSTM. The action distribution path does not consume the PI bottleneck, but the auxiliary loss branches from actor LSTM states and trains the shared recurrent features.

`achieved_goal` must remain in rollout observations as the PI target. `rl-baselines3-zoo/conf/maze_pi.yml` drops it from policy features with `features_extractor_kwargs=dict(drop_keys=['achieved_goal'])`.

```bash
python ./rl-baselines3-zoo/train.py --algo pi_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze_pi.yml
```

## Visualize

### Training Logs

```bash
tensorboard --logdir logs/xxx
```

### Best Policy Rollout

Follow `scripts/sb3_extract_infos.sh` and inspect the command before running it.
