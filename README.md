# MorphNav

## Install

Follow `scripts/build_conda_env.sh`.

## Path Layout

New runs use two top-level roots:

- `data/datasets/` stores reusable datasets that can be shared across experiments.
- `runs/` stores experiment outputs, including SB3 models, TensorBoard logs, rollout recordings, metrics, and analysis artifacts.

Keep dataset analysis and representation analysis under the run that consumes the data.

## PointMaze Offline PI Dataset Generation

The standalone generator writes the current PointMaze dataset schema implemented by `components/dataset_gen/`. Use the Phase 1 preset for offline path-integration rehearsal/probe data. The preset samples trajectories with a grid-cells-style smooth velocity random walk, then uses a PointMaze global `x/y` force controller to track that desired velocity through MuJoCo dynamics. It keeps `sensor_aware=True`, so `obs/observation` includes MuJoCo velocity plus four touch-sensor channels. The collection driver reacts to those touch channels instead of using hard-coded arena boundaries or a synthetic collision flag: contact response uses a tangent-biased direction with a weaker away-from-wall component plus bounded jitter, without random heading resampling in the touch branch. The default storage is compact NPZ shards to avoid directory-style Zarr small-file overhead.

```bash
python scripts/generate_pointmaze_dataset.py \
  --preset phase1_pointmaze_pi \
  --num-episodes 10000 \
  --episodes-per-shard 1000 \
  --dataset-seed 0
```

Use `--dataset-name phase1_pi/probe_seed1 --num-episodes 4000 --dataset-seed 1` for a held-out probe dataset.

By default, the generation command writes to `data/datasets/pointmaze/<dataset-name>/`:

- `manifest.json`
- `dataset_metadata.json`
- `shard_000000.npz`, `shard_000001.npz`, and so on

If `--dataset-name` is omitted for the Phase 1 preset, the generator uses `phase1_pi/rehearsal_seed<dataset-seed>`. Add `--timestamp-name` to append `<YYYYMMDD_HHMMSS>` to that automatic name. Existing shard files are not replaced unless `--overwrite` is passed.

Shard arrays include `episode_lengths`, `episode_offsets`, `step/action`, `step/reward`, `step/terminated`, `step/truncated`, `step/qpos`, `step/qvel`, `step/goal`, `obs/observation`, `obs/start_pos`, `obs/achieved_goal`, `obs/desired_goal`, `annotation/agent_xy`, `annotation/heading`, `annotation/goal_xy`, and `annotation/relative_goal`.

Each shard includes `dataset_meta_json` and `episode_summaries_json` entries. Use `--storage-format zarr` only when directory-style Zarr output is explicitly needed.

`obs/*` rows are action-before policy observations aligned one-to-one with `step/action`. Raw `info` payloads are not persisted. Keep the target `pi_ppo_lstm` RL run sensor-aware as well, or the dataset and policy observation spaces will not match.

Validate dataset coverage before training:

```bash
python scripts/analyze_pointmaze_dataset.py \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0
```

By default, the validation command writes `dataset_distribution.json`, `occupancy.png`, `action_hist.png`, and `trajectory_preview.png` under `runs/offline_pi/pointmaze_phase1_seed0/analysis/datasets/<dataset-root-name>/`. Pass `--output-dir` to override that location.

## Offline PI Rehearsal

Offline rehearsal trains only the PI path of `pi_ppo_lstm` using place-cell cross-entropy. MSE is reported only as a localization metric.

```bash
python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 1
```

By default, the command creates a timestamped run under `runs/offline_pi/pointmaze_phase1_seed<seed>_<YYYYMMDD_HHMMSS>/` and writes `models/final_model.zip` and `metrics/offline_pi_metrics.json` there. Pass `--run-name` for a stable name or `--output-dir` for an explicit path.

Analyze offline PI bottleneck spatial representations:

```bash
python scripts/analyze_offline_pi_representations.py \
  --model-path runs/offline_pi/<run-name>/models/final_model.zip \
  --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1
```

By default, representation analysis writes under `<offline-pi-run>/analysis/representations/<dataset-root-name>/`. Pass `--output-dir` to override that location.

## Train

Follow `scripts/sb3zoo_train.sh` and inspect the command before running it.

```bash
python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --log-folder runs/sb3 --tensorboard-log runs/tensorboard/sb3
```

### Path integration auxiliary training

`pi_ppo_lstm` adds a parallel place-cell prediction branch to PPO-LSTM. The action distribution path does not consume the PI bottleneck, but the auxiliary loss branches from actor LSTM states and trains the shared recurrent features.

`achieved_goal` must remain in rollout observations as the PI target. Phase 1 PointMaze PI runs should use `sensor_aware=True` to match the dataset preset. `rl-baselines3-zoo/conf/maze_pi.yml` drops `achieved_goal` from policy features with `features_extractor_kwargs=dict(drop_keys=['achieved_goal'])`.

```bash
python ./rl-baselines3-zoo/train.py --algo pi_ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze_pi.yml --log-folder runs/sb3 --tensorboard-log runs/tensorboard/sb3
```

## Visualize

### Training Logs

```bash
tensorboard --logdir runs/tensorboard
```

### Best Policy Rollout

Follow `scripts/sb3_extract_infos.sh` and inspect the command before running it.
