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
- Representation analysis: keep the existing `scripts/analyze_offline_pi_representations.py` workflow, but make its outputs easier to associate with a run.

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

`metrics/offline_pi_metrics.json` contains the final training metrics plus the latest probe metrics when a probe dataset is provided.

## Logging Frequency

Add CLI parameters:

- `--log-every-updates`
- `--eval-every-epochs`
- `--eval-at-start`
- `--no-eval-at-start`
- `--eval-artifact-every-epochs`

`--log-every-updates 0` means the script chooses a bounded default that records progress roughly ten times per epoch. Positive values record every N optimizer updates.

`--eval-every-epochs 0` disables periodic probe evaluation after training epochs.

`--eval-at-start` records a baseline probe at epoch 0 before any optimizer update. It is effective only when `--probe-dataset-root` is set.

`--eval-artifact-every-epochs 0` disables image and NPZ evaluation artifacts. Numeric probe metrics can still run.

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

The effective config is written as a TensorBoard text summary named `run/config` when the writer supports text summaries.

If TensorBoard dependencies are unavailable and TensorBoard was explicitly requested, the script should fail with a clear dependency message. If TensorBoard was not requested, JSON and text logging continue without TensorBoard.

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

Suggested responsibilities:

- `scripts/offline_pi_rehearsal.py`: parse CLI, resolve paths, create env/model, call runtime.
- `components/offline_pi_rehearsal.py`: dataset batching, PI loss, PI probe, PI train loop.
- `components/offline_pi_runtime.py`: run directory setup, logger, config writing, JSONL event writing, TensorBoard writer, checkpoint metadata.
- `components/offline_pi_eval_artifacts.py`: coordinate prediction export and diagnostic plots.

Avoid broad refactors of online PPO training code.

## Documentation Updates

When implementation lands, update:

- `README.md`
- `CLAUDE.md` if any project trap or command expectation changes
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
