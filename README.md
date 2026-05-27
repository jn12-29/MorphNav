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

Offline rehearsal trains only the PI path of `pi_ppo_lstm` using place-cell cross-entropy. Full-sequence and first-step MSE are reported only as localization metrics.

Use the checked-in command note for the default Phase 1 rehearsal/probe datasets:

```bash
bash scripts/pi.sh
```

```bash
python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 1 \
  --eval-every-epochs 1 \
  --checkpoint-every-epochs 1
```

By default, the command creates a timestamped run under `runs/offline_pi/pointmaze_phase1_seed<seed>_<YYYYMMDD_HHMMSS>/`. Each run writes `train.log`, `config.json`, `metrics/metrics.jsonl`, `metrics/offline_pi_metrics.json`, probe summaries under `metrics/probe_epoch_XXXX.json`, checkpoints under `models/`, and optional probe diagnostics under `eval/`. Pass `--run-name` for a stable name or `--output-dir` for an explicit path. TensorBoard scalar logging is attempted by default under the run's `tensorboard/` directory; if TensorBoard dependencies are unavailable, training falls back to JSON and text logging. Use `--no-tensorboard` to skip TensorBoard explicitly. Set `--eval-artifact-every-epochs N` to write probe NPZ, JSON, and PNG localization diagnostics every `N` evaluated epochs.

Offline PI metrics include full-sequence `offline_pi/localization_*` and first-step `offline_pi/first_localization_*` fields, plus `first_localization_mse_ratio` and `first_localization_mae_ratio` against the matching full-sequence metric. The first-step target is timestep 0 of each recurrent sequence, not a separate raw `start_pos` target. Probe artifact summaries include the same first-step error summary, and `pred_vs_target_epoch_XXXX.npz` stores `first_step_mask`.

Grid-score analysis during probe eval is opt-in. Set `--eval-gridscore-every-epochs N` to collect PI bottleneck activity on the probe dataset every `N` evaluated epochs and write `eval/gridscore_epoch_XXXX/gridscore_summary.json`, `gridscore_data.npz`, `top_grid_cells.png`, and `spatial_ratemaps_grid.png`. The probe JSON, `metrics.jsonl`, final metrics JSON, and TensorBoard include `offline_pi/probe/gridscore/*` summaries for epochs where the analysis ran. Use `--gridscore-max-steps` to bound training-time cost.

Run an existing checkpoint on a held-out dataset without training updates:

```bash
python scripts/offline_pi_rehearsal.py \
  --mode probe \
  --model-path runs/offline_pi/<run-name>/models/final_model.zip \
  --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --output-dir runs/offline_pi/<run-name>_probe
```

Analyze offline PI bottleneck spatial representations:

```bash
python scripts/analyze_offline_pi_representations.py \
  --model-path runs/offline_pi/<run-name>/models/final_model.zip \
  --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1
```

By default, representation analysis writes under `<offline-pi-run>/analysis/representations/<dataset-root-name>/` and includes `analysis_config.json` with the source model, dataset, and analysis parameters. It uses the same grid-score implementation as probe-time eval. Pass `--output-dir` to override that location.

## Train

Follow `scripts/sb3zoo_train.sh` and inspect the command before running it.

```bash
python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml --log-folder runs/sb3 --tensorboard-log runs/tensorboard/sb3
```

### Path integration auxiliary training

`pi_ppo_lstm` adds a parallel place-cell prediction branch to PPO-LSTM. The action distribution path does not consume the PI bottleneck, but the auxiliary loss branches from actor LSTM states and trains the shared recurrent features.

`achieved_goal` must remain in rollout observations as the PI target. `start_pos` must also remain in observations and per-step policy features, and is encoded through the same fixed place-cell population to initialize actor/critic LSTM states at `episode_start`. Phase 1 PointMaze PI runs should use `sensor_aware=True` to match the dataset preset. `rl-baselines3-zoo/conf/maze_pi.yml` drops only `achieved_goal` from per-step policy features with `features_extractor_kwargs=dict(drop_keys=['achieved_goal'])`; `start_pos` also enters through `pi_init_state_key='start_pos'`.
Standalone offline PI runs create fresh models from the same `rl-baselines3-zoo/conf/maze_pi.yml` model settings.

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
