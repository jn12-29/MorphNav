# Offline PI Training Logging And Evaluation Goal

## Goal

Make standalone PointMaze offline PI rehearsal a fully observable and reproducible experiment workflow.

The workflow must record the effective run configuration, stream useful training progress, evaluate held-out probe performance before and during training, export coordinate-localization diagnostics, and save model checkpoints with enough context for later inspection and future resume support.

## Scope

- Entry point: `scripts/offline_pi_rehearsal.py`.
- Core training and probe helpers: `components/offline_pi_rehearsal.py`.
- Dataset family: `data/datasets/pointmaze/phase1_pi/*`.
- Model family: `PathIntegrationRecurrentPPO` with `PathIntegrationRecurrentActorCriticPolicy`.
- Training objective: keep the existing place-cell cross-entropy objective.
- Localization metrics: report decoded coordinate metrics from place-cell probabilities.
- Representation analysis: keep the standalone `scripts/analyze_offline_pi_representations.py` workflow and add an explicit opt-in path that runs bottleneck grid-score analysis during probe evaluation.

Out of scope:

- Changing the offline PI loss semantics.
- Changing dataset schema or collection behavior.
- Replacing the SB3 model save format.
- Making grid-score, spatial autocorrelogram, head-direction tuning, or shuffle-significance statistics part of default offline PI training.
- Default MP4 generation.

## Run Directory Contract

Each standalone offline PI run writes to one run directory:

```text
runs/offline_pi/<run-name>/
  train.log
  config.json
  metrics/
    metrics.jsonl
    offline_pi_metrics.json
    probe_epoch_0000.json
    probe_epoch_XXXX.json
  tensorboard/
  models/
    final_model.zip
    checkpoint_epoch_XXXX.zip
  eval/
    pred_vs_target_epoch_XXXX.npz
    error_summary_epoch_XXXX.json
    coord_scatter_epoch_XXXX.png
    error_hist_epoch_XXXX.png
    spatial_error_heatmap_epoch_XXXX.png
    gridscore_epoch_XXXX/
      gridscore_summary.json
      gridscore_data.npz
      top_grid_cells.png
      spatial_ratemaps_grid.png
```

`offline_pi_metrics.json` remains the final compact summary for compatibility with existing users and scripts. `metrics.jsonl` is the append-only event stream for progress, epoch summaries, probe summaries, and checkpoint events.

## Effective Config

Every run writes `config.json` before training updates begin.

The config must include:

- CLI arguments after default resolution.
- Output directory and resolved run name.
- Dataset roots:
  - `dataset_root`
  - `probe_dataset_root`
- Training parameters:
  - `learning_rate`
  - `batch_size_sequences`
  - `max_seq_len`
  - `max_updates`
  - `epochs`
  - `seed`
  - `device`
- Logging and evaluation parameters:
  - `log_every_updates`
  - `eval_every_epochs`
  - `eval_at_start`
  - `tensorboard_log_dir`
  - `eval_artifact_every_epochs`
  - `eval_gridscore_every_epochs`
  - `gridscore_n_bins`
  - `gridscore_max_steps`
  - `gridscore_top_k`
  - `checkpoint_every_epochs`
  - `save_final_checkpoint`
- Fresh-model settings loaded from `rl-baselines3-zoo/conf/maze_pi.yml`.
- Dataset metadata summaries for rehearsal and probe roots when metadata files are present.

The file records the final run contract only. It should not include migration notes, stale parameter names, or unused legacy fields.

## Training Log

The run writes `train.log` and mirrors the same high-level messages to stdout.

Required log events:

- run start with output directory, dataset root, probe dataset root, device, seed, epochs, max sequence length, and batch size;
- config write completion;
- baseline probe start and completion when enabled;
- epoch start and completion;
- periodic update progress;
- scheduled probe start and completion;
- checkpoint write completion;
- final model write completion;
- final metrics write completion.

The log must be concise and stable enough to inspect during long runs without reading JSON files.

## Metrics Event Stream

`metrics/metrics.jsonl` stores one JSON object per line.

Required common fields:

- `event`
- `phase`
- `epoch`
- `update`
- `global_step`
- `timestamp`

Required event types:

- `run_start`
- `probe`
- `train_update`
- `train_epoch`
- `checkpoint`
- `run_end`

Training update events include:

- `offline_pi/loss_step`
- `offline_pi/localization_mse_step`
- `offline_pi/masked_steps`
- `offline_pi/sequence_count`
- `offline_pi/lr`

Training epoch events include:

- `offline_pi/loss_mean`
- `offline_pi/loss_std`
- `offline_pi/localization_mse_mean`
- `offline_pi/localization_mse_std`
- `offline_pi/steps`
- `offline_pi/sequence_count`
- `offline_pi/updates`
- `offline_pi/epoch_seconds`
- `offline_pi/samples_seen`

Probe events include:

- `offline_pi/probe/loss`
- `offline_pi/probe/localization_mse`
- `offline_pi/probe/localization_rmse`
- `offline_pi/probe/localization_mae`
- `offline_pi/probe/x_mae`
- `offline_pi/probe/y_mae`
- `offline_pi/probe/steps`
- `offline_pi/probe/sequence_count`
- `offline_pi/probe/seconds`

When grid-score analysis is enabled for a probe event, that event also includes:

- `offline_pi/probe/gridscore/best`
- `offline_pi/probe/gridscore/best_unit`
- `offline_pi/probe/gridscore/mean`
- `offline_pi/probe/gridscore/valid_units`
- `offline_pi/probe/gridscore/seconds`

`metrics/offline_pi_metrics.json` contains the final training metrics plus the latest probe metrics when a probe dataset is provided.

## Logging Frequency

Add CLI parameters:

- `--log-every-updates`
- `--eval-every-epochs`
- `--eval-at-start`
- `--no-eval-at-start`
- `--eval-artifact-every-epochs`
- `--eval-gridscore-every-epochs`
- `--gridscore-n-bins`
- `--gridscore-max-steps`
- `--gridscore-top-k`

`--log-every-updates 0` means the script chooses a bounded default that records progress roughly ten times per epoch. Positive values record every N optimizer updates.

`--eval-every-epochs 0` disables periodic probe evaluation after training epochs.

`--eval-at-start` records a baseline probe at epoch 0 before any optimizer update. It is effective only when `--probe-dataset-root` is set.

`--eval-artifact-every-epochs 0` disables image and NPZ evaluation artifacts. Numeric probe metrics can still run.

`--eval-gridscore-every-epochs 0` disables grid-score analysis during training and probe evaluation. Positive values run bottleneck grid-score analysis on the probe dataset every N evaluated epochs. The epoch 0 baseline probe is eligible when `--eval-at-start` is enabled and the value is positive.

Grid-score analysis defaults to `--gridscore-n-bins 32`, `--gridscore-top-k 8`, and no step cap unless `--gridscore-max-steps` is provided. Probe-time grid-score eval analyzes all bottleneck units.

## Baseline And Periodic Probe

When `--probe-dataset-root` is set and `--eval-at-start` is enabled, the workflow evaluates the fresh or loaded model before training updates and writes:

```text
metrics/probe_epoch_0000.json
```

After each completed epoch, if `epoch % eval_every_epochs == 0`, the workflow evaluates the probe dataset again and writes:

```text
metrics/probe_epoch_XXXX.json
```

Probe evaluation never updates model parameters.

Probe metrics use the same decoded coordinate calculation as training metrics so that train and probe localization metrics are directly comparable.

## Probe Grid-Score Analysis

When grid-score analysis is enabled for a selected probe epoch, the workflow collects PI bottleneck activity on the probe dataset, computes spatial rate maps, computes 2D autocorrelograms, and reports grid-score summaries without updating model parameters.

Each selected epoch writes:

```text
eval/gridscore_epoch_XXXX/
  gridscore_summary.json
  gridscore_data.npz
  top_grid_cells.png
  spatial_ratemaps_grid.png
```

`gridscore_summary.json` contains:

- `best_grid_score`
- `best_unit`
- `mean_grid_score`
- `valid_units`
- `unit_count`
- `num_steps`
- `bounds`
- `n_bins`
- `max_steps`
- `top_k`

`gridscore_data.npz` contains:

- `positions`
- `activations`
- `ratemaps`
- `autocorrs`
- `grid_scores`
- `bounds`

The probe JSON, JSONL probe event, final metrics JSON, and TensorBoard scalars include grid-score summaries only for epochs where grid-score analysis ran. If no valid grid scores are available, numeric summary fields are stored as `NaN` and `valid_units` is `0`.

## TensorBoard

TensorBoard logging is optional and controlled by CLI.

Add CLI parameters:

- `--tensorboard`
- `--tensorboard-log-dir`

When enabled, scalar summaries are written under:

```text
runs/offline_pi/<run-name>/tensorboard/
```

Required scalar tags:

- `train/loss_step`
- `train/localization_mse_step`
- `train/loss_epoch`
- `train/localization_mse_epoch`
- `train/lr`
- `train/epoch_seconds`
- `probe/loss`
- `probe/localization_mse`
- `probe/localization_rmse`
- `probe/localization_mae`
- `probe/x_mae`
- `probe/y_mae`
- `probe/gridscore/best`
- `probe/gridscore/mean`
- `probe/gridscore/valid_units`
- `probe/gridscore/seconds`

The effective config is written as a TensorBoard text summary named `run/config` when the writer supports text summaries.

If TensorBoard dependencies are unavailable, the run records the disabled reason in `config.json` and continues with JSON and text logging.

## Coordinate Prediction Helper

Add a shared helper for decoding place-cell logits into coordinate predictions.

The helper should return:

- `pred_xy`
- `target_xy`
- `mask`
- aggregate metrics:
  - `mse`
  - `rmse`
  - `mae`
  - `x_mae`
  - `y_mae`

`compute_offline_pi_loss(...)` and probe artifact export should use this helper instead of duplicating place-cell decoding logic.

## Evaluation Artifacts

When probe artifact export is enabled, each selected probe epoch writes:

```text
eval/pred_vs_target_epoch_XXXX.npz
eval/error_summary_epoch_XXXX.json
eval/coord_scatter_epoch_XXXX.png
eval/error_hist_epoch_XXXX.png
eval/spatial_error_heatmap_epoch_XXXX.png
```

`pred_vs_target_epoch_XXXX.npz` contains:

- `pred_xy`
- `target_xy`
- `mask`
- `squared_error`
- `absolute_error`
- `bounds`

`error_summary_epoch_XXXX.json` contains:

- `mse`
- `rmse`
- `mae`
- `x_mae`
- `y_mae`
- `max_error`
- `p50_error`
- `p90_error`
- `p95_error`
- `num_steps`
- `bounds`

The scatter plot compares predicted xy against target xy. The histogram plots Euclidean localization error. The spatial heatmap bins target positions and shows mean Euclidean error per bin.

Evaluation artifacts are written only for probe datasets, not for training batches.

Grid-score artifacts are also written only for probe datasets and only when `--eval-gridscore-every-epochs` selects that probe epoch.

## Checkpoints

Add CLI parameters:

- `--checkpoint-every-epochs`
- `--save-final-checkpoint`
- `--no-save-final-checkpoint`

Checkpoint files are written under:

```text
models/checkpoint_epoch_XXXX.zip
models/final_model.zip
```

The existing SB3 zip format remains the model artifact. A sidecar metadata file is written next to each checkpoint:

```text
models/checkpoint_epoch_XXXX.json
models/final_model.json
```

Checkpoint metadata includes:

- `schema_version`
- `epoch`
- `update`
- `global_step`
- `config`
- `dataset_info`
- `probe_dataset_info`
- `latest_train_metrics`
- `latest_probe_metrics`
- `model_path`
- `created_at`

Writes should be atomic where practical: write to a temporary path in the same directory, then replace the target path.

The first implementation may be save-only. Metadata should still be structured so future resume support can add optimizer and scheduler state without changing the top-level checkpoint identity.

## Code Organization

Keep the CLI entry point thin.

Suggested helper modules:

- `components/offline_pi_runtime.py`
- `components/offline_pi_eval_artifacts.py`
- `components/offline_pi_gridscore.py`

Suggested responsibilities:

- `scripts/offline_pi_rehearsal.py`: parse CLI, resolve paths, create env/model, call runtime.
- `components/offline_pi_rehearsal.py`: dataset batching, PI loss, PI probe, PI train loop.
- `components/offline_pi_runtime.py`: run directory setup, logger, config writing, JSONL event writing, TensorBoard writer, checkpoint metadata.
- `components/offline_pi_eval_artifacts.py`: coordinate prediction export and diagnostic plots.
- `components/offline_pi_gridscore.py`: bottleneck activity collection, spatial ratemap computation, autocorrelogram/grid-score metrics, and grid-score artifact export.

Avoid broad refactors of online PPO training code.

## Documentation Updates

When implementation lands, update:

- `README.md`
- `AGENTS.md` if any project trap or command expectation changes
- relevant `scripts/*.sh` experiment notes if they invoke offline PI training

Documentation should present the final workflow and current commands only. It should not preserve old command variants as migration notes unless explicitly requested.

## Acceptance Criteria

The work is complete when:

- A train run writes `train.log`, `config.json`, `metrics/metrics.jsonl`, `metrics/offline_pi_metrics.json`, and `models/final_model.zip`.
- A train run with `--probe-dataset-root` and baseline eval writes `metrics/probe_epoch_0000.json`.
- Periodic probe writes epoch-numbered probe JSON files according to `--eval-every-epochs`.
- Update and epoch metrics are visible during training in stdout and `train.log`.
- TensorBoard scalar logging works when explicitly enabled.
- Probe artifact export writes NPZ, JSON, and PNG diagnostics according to `--eval-artifact-every-epochs`.
- Probe grid-score analysis writes summary JSON, NPZ data, and PNG diagnostics according to `--eval-gridscore-every-epochs`.
- Probe grid-score summaries appear in `metrics.jsonl`, `offline_pi_metrics.json`, and TensorBoard only for selected probe epochs.
- Checkpoint files and sidecar metadata are written according to `--checkpoint-every-epochs`.
- Existing final metrics JSON remains available at `metrics/offline_pi_metrics.json`.
- Probe mode still works without training updates.
- The smallest relevant tests and smoke commands pass.

## Validation Commands

Use focused validation first:

```bash
conda run -n mz python -m pytest tests/test_offline_pi_rehearsal.py
```

Run a short smoke train:

```bash
conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode train \
  --dataset-root data/datasets/pointmaze/phase1_pi/rehearsal_seed0 \
  --probe-dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --epochs 1 \
  --max-updates 2 \
  --run-name offline_pi_logging_eval_smoke \
  --log-every-updates 1 \
  --eval-every-epochs 1 \
  --eval-artifact-every-epochs 1 \
  --eval-gridscore-every-epochs 1 \
  --gridscore-max-steps 256 \
  --checkpoint-every-epochs 1
```

Run a probe smoke:

```bash
conda run -n mz python scripts/offline_pi_rehearsal.py \
  --mode probe \
  --model-path runs/offline_pi/offline_pi_logging_eval_smoke/models/final_model.zip \
  --dataset-root data/datasets/pointmaze/phase1_pi/probe_seed1 \
  --output-dir runs/offline_pi/offline_pi_logging_eval_smoke_probe
```

Verify expected files:

```bash
find runs/offline_pi/offline_pi_logging_eval_smoke -maxdepth 3 -type f | sort
```
