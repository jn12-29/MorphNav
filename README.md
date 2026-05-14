# MorphNav

## Install

Follow `scripts/build_conda_env.sh`.

## PointMaze Offline PI Dataset Generation

The standalone generator writes the current PointMaze dataset schema implemented by `components/dataset_gen/`. Use the Phase 1 preset for offline path-integration rehearsal/probe data. The preset samples trajectories with a grid-cells-style smooth velocity random walk, then uses a PointMaze global `x/y` force controller to track that desired velocity through MuJoCo dynamics. The default storage is compact NPZ shards to avoid directory-style Zarr small-file overhead.

```bash
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --output-dir recorded_data \
  --dataset-name pointmaze_mujoco_pi_rehearsal \
  --num-episodes 10000 \
  --episodes-per-shard 1000 \
  --dataset-seed 0
```

Use `--dataset-name pointmaze_mujoco_pi_probe --num-episodes 4000 --dataset-seed 1` for a held-out probe dataset.

The generation command writes to `<output-dir>/<dataset-name>/`:

- `manifest.json`
- `dataset_metadata.json`
- `shard_000000.npz`, `shard_000001.npz`, and so on

Shard arrays include `episode_lengths`, `episode_offsets`, `step/action`, `step/reward`, `step/terminated`, `step/truncated`, `step/qpos`, `step/qvel`, `step/goal`, `obs/observation`, `obs/start_pos`, `obs/achieved_goal`, `obs/desired_goal`, `annotation/agent_xy`, `annotation/heading`, `annotation/goal_xy`, and `annotation/relative_goal`.

Each shard includes `dataset_meta_json` and `episode_summaries_json` entries. Use `--storage-format zarr` only when directory-style Zarr output is explicitly needed.

`obs/*` rows are action-before policy observations aligned one-to-one with `step/action`. Raw `info` payloads are not persisted.

Validate dataset coverage before training:

```bash
python scripts/analyze_pointmaze_dataset.py \
  --dataset-root recorded_data/pointmaze_mujoco_pi_rehearsal \
  --output-dir logs/offline_pi/pointmaze_phase1_seed0/dataset_analysis
```

The validation command writes `dataset_distribution.json`, `occupancy.png`, `action_hist.png`, and `trajectory_preview.png` directly under its `--output-dir`.

## Offline PI Rehearsal

Offline rehearsal trains only the PI path of `pi_ppo_lstm` using place-cell cross-entropy. MSE is reported only as a localization metric.

```bash
python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root recorded_data/pointmaze_mujoco_pi_rehearsal \
  --probe-dataset-root recorded_data/pointmaze_mujoco_pi_probe \
  --output-dir logs/offline_pi/pointmaze_phase1_seed0 \
  --epochs 1
```

Analyze offline PI bottleneck spatial representations:

```bash
python scripts/analyze_offline_pi_representations.py \
  --model-path logs/offline_pi/pointmaze_phase1_seed0/final_model.zip \
  --dataset-root recorded_data/pointmaze_mujoco_pi_probe \
  --output-dir logs/offline_pi/pointmaze_phase1_seed0/analysis
```

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
