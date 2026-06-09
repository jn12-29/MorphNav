"""Offline PointMaze PI rehearsal and probe helpers.

This module trains or evaluates only the PI path of the current `pi_ppo_lstm`
model. The optimization loss is weighted place-cell cross-entropy; coordinate
MSE is reported only as a localization metric, and offline rehearsal uses an
optimizer separate from PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch as th
import zarr

from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
    POINTMAZE_POLICY_OBS_KEYS,
)
from components.path_integration import recurrent_first_step_loss_weights, soft_place_cell_cross_entropy
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


_METRIC_RATIO_EPS = 1e-12
OFFLINE_PI_OPTIMIZER_CLASSES: dict[str, type[th.optim.Optimizer]] = {
    "adam": th.optim.Adam,
    "adamw": th.optim.AdamW,
    "rmsprop": th.optim.RMSprop,
    "sgd": th.optim.SGD,
}


def _metric_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), _METRIC_RATIO_EPS)


def offline_pi_optimizer_names() -> tuple[str, ...]:
    return tuple(OFFLINE_PI_OPTIMIZER_CLASSES)


def resolve_offline_pi_optimizer_class(name: str) -> type[th.optim.Optimizer]:
    key = str(name).lower()
    if key not in OFFLINE_PI_OPTIMIZER_CLASSES:
        valid = ", ".join(offline_pi_optimizer_names())
        raise ValueError(f"unsupported offline PI optimizer {name!r}; expected one of: {valid}")
    return OFFLINE_PI_OPTIMIZER_CLASSES[key]


@dataclass(frozen=True)
class OfflinePIBatch:
    obs: dict[str, th.Tensor]
    target_pos: th.Tensor
    episode_starts: th.Tensor
    mask: th.Tensor
    lstm_states_pi: tuple[th.Tensor, th.Tensor]
    sequence_count: int
    max_len: int


@dataclass(frozen=True)
class OfflinePICoordinatePredictions:
    pred_xy: th.Tensor
    target_xy: th.Tensor
    mask: th.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class _OfflinePISequence:
    obs: dict[str, np.ndarray]
    target_pos: np.ndarray
    length: int


def _resolve_device(device: str | th.device) -> th.device:
    if str(device) == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def _validate_shard_schema_values(schema: Any, version: Any, shard_path: Path) -> None:
    if schema != POINTMAZE_MUJOCO_ZARR_SCHEMA or int(version or -1) != POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION:
        raise ValueError(
            f"{shard_path} has unsupported dataset schema {schema!r} version {version!r}; "
            f"expected {POINTMAZE_MUJOCO_ZARR_SCHEMA!r} version {POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION}"
        )


def _validate_shard_schema(root: zarr.Group, shard_path: Path) -> None:
    _validate_shard_schema_values(root.attrs.get("dataset_schema"), root.attrs.get("dataset_schema_version"), shard_path)


def _shard_paths(dataset_root: Path) -> list[Path]:
    shard_paths = sorted(dataset_root.glob("shard_*.npz")) + sorted(dataset_root.glob("shard_*.zarr"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz or shard_*.zarr files found under {dataset_root}")
    return shard_paths


def _load_zarr_shard(shard_path: Path, obs_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    root = zarr.open_group(str(shard_path), mode="r")
    _validate_shard_schema(root, shard_path)
    lengths = np.asarray(root["episode_lengths"][:], dtype=np.int64)
    offsets = np.asarray(root["episode_offsets"][:], dtype=np.int64)
    obs_arrays = {key: np.asarray(root[f"obs/{key}"][:], dtype=np.float32) for key in obs_keys}
    return lengths, offsets, obs_arrays


def _load_npz_shard(shard_path: Path, obs_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(shard_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["dataset_meta_json"]))
        _validate_shard_schema_values(metadata.get("dataset_schema"), metadata.get("dataset_schema_version"), shard_path)
        lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        obs_arrays = {key: np.asarray(data[f"obs/{key}"], dtype=np.float32) for key in obs_keys}
    return lengths, offsets, obs_arrays


def _load_shard_sequences(shard_path: Path, obs_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if shard_path.suffix == ".npz":
        return _load_npz_shard(shard_path, obs_keys)
    return _load_zarr_shard(shard_path, obs_keys)


def _load_shard_lengths(shard_path: Path) -> np.ndarray:
    if shard_path.suffix == ".npz":
        with np.load(shard_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["dataset_meta_json"]))
            _validate_shard_schema_values(metadata.get("dataset_schema"), metadata.get("dataset_schema_version"), shard_path)
            return np.asarray(data["episode_lengths"], dtype=np.int64)

    root = zarr.open_group(str(shard_path), mode="r")
    _validate_shard_schema(root, shard_path)
    return np.asarray(root["episode_lengths"][:], dtype=np.int64)


def _load_sequences(dataset_root: str | Path, obs_keys: tuple[str, ...], target_key: str, max_seq_len: int | None) -> list[_OfflinePISequence]:
    dataset_root = Path(dataset_root)

    sequences: list[_OfflinePISequence] = []
    for shard_path in _shard_paths(dataset_root):
        lengths, offsets, obs_arrays = _load_shard_sequences(shard_path, obs_keys)
        if target_key not in obs_arrays:
            raise KeyError(f"target_key {target_key!r} is not available in obs arrays")
        if obs_arrays[target_key].shape[-1] < 2:
            raise ValueError(f"obs/{target_key} must have at least 2 coordinates, got {obs_arrays[target_key].shape}")

        for offset, length in zip(offsets, lengths, strict=True):
            episode_start = int(offset)
            episode_end = episode_start + int(length)
            window = max_seq_len or int(length)
            for start in range(episode_start, episode_end, window):
                end = min(start + window, episode_end)
                seq_obs = {key: value[start:end] for key, value in obs_arrays.items()}
                target_pos = seq_obs[target_key][..., :2]
                if "start_pos" in seq_obs and seq_obs["start_pos"].shape[-1] >= 2:
                    seq_obs["start_pos"] = np.repeat(target_pos[:1], end - start, axis=0).astype(np.float32, copy=False)
                sequences.append(_OfflinePISequence(obs=seq_obs, target_pos=target_pos, length=end - start))

    return sequences


def count_offline_pi_sequences(dataset_root: str | Path, *, max_seq_len: int | None = None) -> int:
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive when provided")

    window = max_seq_len
    count = 0
    for shard_path in _shard_paths(Path(dataset_root)):
        lengths = _load_shard_lengths(shard_path)
        if window is None:
            count += int(lengths.shape[0])
        else:
            count += int(np.ceil(lengths.astype(np.float64) / float(window)).sum())
    return count


def _make_batch(
    sequences: list[_OfflinePISequence],
    *,
    device: th.device,
    n_lstm_layers: int,
    lstm_hidden_size: int,
) -> OfflinePIBatch:
    if not sequences:
        raise ValueError("sequences must be non-empty")

    sequence_count = len(sequences)
    max_len = max(seq.length for seq in sequences)
    obs: dict[str, th.Tensor] = {}
    for key in sequences[0].obs:
        sample_shape = sequences[0].obs[key].shape[1:]
        padded = np.zeros((sequence_count, max_len, *sample_shape), dtype=np.float32)
        for seq_idx, seq in enumerate(sequences):
            padded[seq_idx, : seq.length] = seq.obs[key]
        obs[key] = th.as_tensor(padded.reshape(sequence_count * max_len, *sample_shape), device=device)

    target = np.zeros((sequence_count, max_len, 2), dtype=np.float32)
    episode_starts = np.zeros((sequence_count, max_len), dtype=np.float32)
    mask = np.zeros((sequence_count, max_len), dtype=bool)
    for seq_idx, seq in enumerate(sequences):
        target[seq_idx, : seq.length] = seq.target_pos
        episode_starts[seq_idx, 0] = 1.0
        mask[seq_idx, : seq.length] = True

    h0 = th.zeros((n_lstm_layers, sequence_count, lstm_hidden_size), dtype=th.float32, device=device)
    c0 = th.zeros_like(h0)
    return OfflinePIBatch(
        obs=obs,
        target_pos=th.as_tensor(target.reshape(sequence_count * max_len, 2), device=device),
        episode_starts=th.as_tensor(episode_starts.reshape(sequence_count * max_len), device=device),
        mask=th.as_tensor(mask.reshape(sequence_count * max_len), device=device),
        lstm_states_pi=(h0, c0),
        sequence_count=sequence_count,
        max_len=max_len,
    )


def first_step_mask_from_batch(batch: OfflinePIBatch) -> th.Tensor:
    first_step_mask = th.zeros_like(batch.mask, dtype=th.bool)
    if batch.sequence_count <= 0 or batch.max_len <= 0:
        return first_step_mask
    offsets = th.arange(batch.sequence_count, device=batch.mask.device) * int(batch.max_len)
    first_step_mask[offsets] = batch.mask[offsets].bool()
    return first_step_mask


def load_offline_pi_batches(
    dataset_root: str | Path,
    *,
    batch_size_sequences: int,
    max_seq_len: int | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    device: str | th.device = "auto",
    obs_keys: tuple[str, ...] = POINTMAZE_POLICY_OBS_KEYS,
    target_key: str = "achieved_goal",
    n_lstm_layers: int,
    lstm_hidden_size: int,
) -> Iterator[OfflinePIBatch]:
    if batch_size_sequences <= 0:
        raise ValueError("batch_size_sequences must be positive")
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive when provided")

    sequences = _load_sequences(dataset_root, obs_keys, target_key, max_seq_len)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sequences))
        sequences = [sequences[idx] for idx in order]

    resolved_device = _resolve_device(device)
    for start in range(0, len(sequences), batch_size_sequences):
        yield _make_batch(
            sequences[start : start + batch_size_sequences],
            device=resolved_device,
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
        )


def _aggregate_coordinate_metrics(
    pred_xy: th.Tensor,
    target_xy: th.Tensor,
    mask: th.Tensor,
) -> dict[str, float]:
    valid = mask.bool()
    if not th.any(valid):
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "x_mae": 0.0, "y_mae": 0.0}

    diff = pred_xy[valid] - target_xy[valid].to(dtype=pred_xy.dtype)
    per_step_mse = diff.square().mean(dim=-1)
    abs_diff = diff.abs()
    mse = per_step_mse.mean()
    return {
        "mse": float(mse.detach().cpu().item()),
        "rmse": float(th.sqrt(mse).detach().cpu().item()),
        "mae": float(abs_diff.mean().detach().cpu().item()),
        "x_mae": float(abs_diff[:, 0].mean().detach().cpu().item()),
        "y_mae": float(abs_diff[:, 1].mean().detach().cpu().item()),
    }


def decode_offline_pi_coordinates(
    policy: PathIntegrationRecurrentActorCriticPolicy,
    pc_logits: th.Tensor,
    target_xy: th.Tensor,
    mask: th.Tensor,
) -> OfflinePICoordinatePredictions:
    probs = th.softmax(pc_logits, dim=-1)
    centers = policy.path_integration_target_encoder.centers.to(device=probs.device, dtype=probs.dtype)
    pred_xy = probs @ centers
    target_xy = target_xy[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    mask = mask.to(device=pred_xy.device).bool()
    return OfflinePICoordinatePredictions(
        pred_xy=pred_xy,
        target_xy=target_xy,
        mask=mask,
        metrics=_aggregate_coordinate_metrics(pred_xy, target_xy, mask),
    )


def compute_offline_pi_loss(
    policy: PathIntegrationRecurrentActorCriticPolicy,
    batch: OfflinePIBatch,
    *,
    first_step_loss_weight: float = 10.0,
) -> tuple[th.Tensor, dict[str, float]]:
    pi_outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
    pc_targets = policy.path_integration_target_encoder(batch.target_pos).to(dtype=pi_outputs.pc_logits.dtype)
    loss_weights = recurrent_first_step_loss_weights(
        batch.mask,
        sequence_count=batch.sequence_count,
        first_step_weight=first_step_loss_weight,
    )
    loss = soft_place_cell_cross_entropy(pi_outputs.pc_logits, pc_targets, mask=batch.mask, weights=loss_weights)
    loss_weight_sum = (
        loss_weights.to(dtype=pi_outputs.pc_logits.dtype) * batch.mask.to(dtype=pi_outputs.pc_logits.dtype)
    ).sum()

    with th.no_grad():
        decoded = decode_offline_pi_coordinates(policy, pi_outputs.pc_logits, batch.target_pos, batch.mask)
        first_step_mask = first_step_mask_from_batch(batch)
        first_step_metrics = _aggregate_coordinate_metrics(
            decoded.pred_xy,
            decoded.target_xy,
            first_step_mask,
        )

    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "localization_mse": float(decoded.metrics["mse"]),
        "localization_rmse": float(decoded.metrics["rmse"]),
        "localization_mae": float(decoded.metrics["mae"]),
        "x_mae": float(decoded.metrics["x_mae"]),
        "y_mae": float(decoded.metrics["y_mae"]),
        "first_localization_mse": float(first_step_metrics["mse"]),
        "first_localization_rmse": float(first_step_metrics["rmse"]),
        "first_localization_mae": float(first_step_metrics["mae"]),
        "first_x_mae": float(first_step_metrics["x_mae"]),
        "first_y_mae": float(first_step_metrics["y_mae"]),
        "first_localization_mse_ratio": _metric_ratio(
            first_step_metrics["mse"],
            decoded.metrics["mse"],
        ),
        "first_localization_mae_ratio": _metric_ratio(
            first_step_metrics["mae"],
            decoded.metrics["mae"],
        ),
        "loss_weight_sum": float(loss_weight_sum.detach().cpu().item()),
        "masked_steps": float(batch.mask.sum().detach().cpu().item()),
        "first_step_count": float(first_step_mask.sum().detach().cpu().item()),
        "sequence_count": float(batch.sequence_count),
    }


def _unique_trainable_parameters(modules: list[th.nn.Module | None]) -> list[th.nn.Parameter]:
    params: list[th.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def make_offline_pi_optimizer(
    policy: PathIntegrationRecurrentActorCriticPolicy,
    *,
    lr: float,
    optimizer_cls: type[th.optim.Optimizer] = th.optim.Adam,
    **optimizer_kwargs: Any,
) -> th.optim.Optimizer:
    params = _unique_trainable_parameters(
        [
            policy.pi_features_extractor,
            policy.lstm_actor,
            policy.path_integration_state_init,
            policy.path_integration_cell_init,
            policy.path_integration_head,
        ]
    )
    if not params:
        raise ValueError("offline PI optimizer has no trainable parameters")
    return optimizer_cls(params, lr=lr, **optimizer_kwargs)


def _policy_lstm_shape(policy: PathIntegrationRecurrentActorCriticPolicy) -> tuple[int, int]:
    n_lstm_layers, _n_envs, lstm_hidden_size = policy.lstm_hidden_state_shape
    return int(n_lstm_layers), int(lstm_hidden_size)


def _mean_metrics(metric_rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not metric_rows:
        return {
            f"{prefix}/loss": float("nan"),
            f"{prefix}/localization_mse": float("nan"),
            f"{prefix}/localization_rmse": float("nan"),
            f"{prefix}/localization_mae": float("nan"),
            f"{prefix}/x_mae": float("nan"),
            f"{prefix}/y_mae": float("nan"),
            f"{prefix}/first_localization_mse": float("nan"),
            f"{prefix}/first_localization_rmse": float("nan"),
            f"{prefix}/first_localization_mae": float("nan"),
            f"{prefix}/first_x_mae": float("nan"),
            f"{prefix}/first_y_mae": float("nan"),
            f"{prefix}/first_localization_mse_ratio": float("nan"),
            f"{prefix}/first_localization_mae_ratio": float("nan"),
            f"{prefix}/loss_weight_sum": 0.0,
            f"{prefix}/steps": 0.0,
            f"{prefix}/first_step_count": 0.0,
            f"{prefix}/sequence_count": 0.0,
        }
    weighted_loss_steps = sum(row["loss_weight_sum"] for row in metric_rows)
    weighted_steps = sum(row["masked_steps"] for row in metric_rows)
    weighted_first_steps = sum(row["first_step_count"] for row in metric_rows)
    if weighted_loss_steps > 0:
        loss = sum(row["loss"] * row["loss_weight_sum"] for row in metric_rows) / weighted_loss_steps
    else:
        loss = float(np.mean([row["loss"] for row in metric_rows]))
    if weighted_steps > 0:
        localization_mse = sum(row["localization_mse"] * row["masked_steps"] for row in metric_rows) / weighted_steps
        localization_mae = sum(row["localization_mae"] * row["masked_steps"] for row in metric_rows) / weighted_steps
        x_mae = sum(row["x_mae"] * row["masked_steps"] for row in metric_rows) / weighted_steps
        y_mae = sum(row["y_mae"] * row["masked_steps"] for row in metric_rows) / weighted_steps
    else:
        localization_mse = float(np.mean([row["localization_mse"] for row in metric_rows]))
        localization_mae = float(np.mean([row["localization_mae"] for row in metric_rows]))
        x_mae = float(np.mean([row["x_mae"] for row in metric_rows]))
        y_mae = float(np.mean([row["y_mae"] for row in metric_rows]))
    if weighted_first_steps > 0:
        first_localization_mse = sum(
            row["first_localization_mse"] * row["first_step_count"] for row in metric_rows
        ) / weighted_first_steps
        first_localization_mae = sum(
            row["first_localization_mae"] * row["first_step_count"] for row in metric_rows
        ) / weighted_first_steps
        first_x_mae = sum(
            row["first_x_mae"] * row["first_step_count"] for row in metric_rows
        ) / weighted_first_steps
        first_y_mae = sum(
            row["first_y_mae"] * row["first_step_count"] for row in metric_rows
        ) / weighted_first_steps
    else:
        first_localization_mse = float(np.mean([row["first_localization_mse"] for row in metric_rows]))
        first_localization_mae = float(np.mean([row["first_localization_mae"] for row in metric_rows]))
        first_x_mae = float(np.mean([row["first_x_mae"] for row in metric_rows]))
        first_y_mae = float(np.mean([row["first_y_mae"] for row in metric_rows]))
    localization_rmse = float(np.sqrt(localization_mse))
    first_localization_rmse = float(np.sqrt(first_localization_mse))
    return {
        f"{prefix}/loss": float(loss),
        f"{prefix}/localization_mse": float(localization_mse),
        f"{prefix}/localization_rmse": float(localization_rmse),
        f"{prefix}/localization_mae": float(localization_mae),
        f"{prefix}/x_mae": float(x_mae),
        f"{prefix}/y_mae": float(y_mae),
        f"{prefix}/first_localization_mse": float(first_localization_mse),
        f"{prefix}/first_localization_rmse": float(first_localization_rmse),
        f"{prefix}/first_localization_mae": float(first_localization_mae),
        f"{prefix}/first_x_mae": float(first_x_mae),
        f"{prefix}/first_y_mae": float(first_y_mae),
        f"{prefix}/first_localization_mse_ratio": _metric_ratio(
            first_localization_mse,
            localization_mse,
        ),
        f"{prefix}/first_localization_mae_ratio": _metric_ratio(
            first_localization_mae,
            localization_mae,
        ),
        f"{prefix}/loss_weight_sum": float(weighted_loss_steps),
        f"{prefix}/steps": float(weighted_steps),
        f"{prefix}/first_step_count": float(weighted_first_steps),
        f"{prefix}/sequence_count": float(sum(row["sequence_count"] for row in metric_rows)),
    }


def _record_metrics(logger: Any | None, metrics: dict[str, float]) -> None:
    if logger is None:
        return
    record = getattr(logger, "record", None)
    if callable(record):
        for key, value in metrics.items():
            record(key, value)


def _optimizer_params(optimizer: th.optim.Optimizer) -> list[th.nn.Parameter]:
    return [param for group in optimizer.param_groups for param in group["params"]]


def run_offline_pi_rehearsal(
    model: Any,
    dataset_root: str | Path,
    *,
    optimizer: th.optim.Optimizer | None = None,
    optimizer_cls: type[th.optim.Optimizer] = th.optim.Adam,
    optimizer_kwargs: dict[str, Any] | None = None,
    lr: float = 1e-4,
    batch_size_sequences: int = 16,
    max_seq_len: int | None = None,
    max_updates: int | None = None,
    n_epochs: int = 1,
    shuffle: bool = True,
    seed: int | None = None,
    max_grad_norm: float = 0.5,
    first_step_loss_weight: float = 10.0,
    logger: Any | None = None,
    on_epoch_start: Callable[[dict[str, Any]], None] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    if max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative")
    if first_step_loss_weight <= 0.0:
        raise ValueError("first_step_loss_weight must be positive")
    policy: PathIntegrationRecurrentActorCriticPolicy = model.policy
    policy.set_training_mode(True)
    if optimizer is policy.optimizer:
        raise ValueError("offline PI rehearsal must use an optimizer separate from policy.optimizer")
    offline_optimizer = optimizer or make_offline_pi_optimizer(
        policy,
        lr=lr,
        optimizer_cls=optimizer_cls,
        **(optimizer_kwargs or {}),
    )
    offline_optimizer_params = _optimizer_params(offline_optimizer)
    n_lstm_layers, lstm_hidden_size = _policy_lstm_shape(policy)

    metric_rows: list[dict[str, float]] = []
    latest_epoch_metrics: dict[str, float] | None = None
    updates = 0
    for epoch in range(n_epochs):
        policy.set_training_mode(True)
        epoch_number = epoch + 1
        epoch_start = time.time()
        if on_epoch_start is not None:
            on_epoch_start({"epoch": epoch_number, "update": updates})
        epoch_rows: list[dict[str, float]] = []
        batches = load_offline_pi_batches(
            dataset_root,
            batch_size_sequences=batch_size_sequences,
            max_seq_len=max_seq_len,
            shuffle=shuffle,
            seed=None if seed is None else seed + epoch,
            device=policy.device,
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
        )
        for batch in batches:
            offline_optimizer.zero_grad()
            loss, metrics = compute_offline_pi_loss(
                policy,
                batch,
                first_step_loss_weight=first_step_loss_weight,
            )
            loss.backward()
            if max_grad_norm > 0.0:
                th.nn.utils.clip_grad_norm_(offline_optimizer_params, max_grad_norm)
            offline_optimizer.step()
            metric_rows.append(metrics)
            epoch_rows.append(metrics)
            updates += 1
            if on_update is not None:
                on_update(
                    {
                        "epoch": epoch_number,
                        "update": updates,
                        "metrics": metrics,
                        "lr": float(offline_optimizer.param_groups[0]["lr"]),
                    }
                )
            if max_updates is not None and updates >= max_updates:
                break
        latest_epoch_metrics = _mean_metrics(epoch_rows, "offline_pi")
        latest_epoch_metrics["offline_pi/updates"] = float(len(epoch_rows))
        latest_epoch_metrics["offline_pi/epoch_seconds"] = float(time.time() - epoch_start)
        latest_epoch_metrics["offline_pi/samples_seen"] = float(
            sum(row["masked_steps"] for row in epoch_rows)
        )
        if epoch_rows:
            latest_epoch_metrics["offline_pi/loss_std"] = float(np.std([row["loss"] for row in epoch_rows]))
            latest_epoch_metrics["offline_pi/localization_mse_std"] = float(
                np.std([row["localization_mse"] for row in epoch_rows])
            )
            latest_epoch_metrics["offline_pi/first_localization_mse_std"] = float(
                np.std([row["first_localization_mse"] for row in epoch_rows])
            )
        else:
            latest_epoch_metrics["offline_pi/loss_std"] = float("nan")
            latest_epoch_metrics["offline_pi/localization_mse_std"] = float("nan")
            latest_epoch_metrics["offline_pi/first_localization_mse_std"] = float("nan")
        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch_number,
                    "update": updates,
                    "metrics": latest_epoch_metrics,
                    "epoch_seconds": latest_epoch_metrics["offline_pi/epoch_seconds"],
                }
            )
        if max_updates is not None and updates >= max_updates:
            break

    metrics = _mean_metrics(metric_rows, "offline_pi")
    metrics["offline_pi/updates"] = float(updates)
    if latest_epoch_metrics is not None:
        metrics["offline_pi/last_epoch_seconds"] = float(latest_epoch_metrics["offline_pi/epoch_seconds"])
        metrics["offline_pi/final_epoch"] = float(epoch_number)
    _record_metrics(logger, metrics)
    return metrics


@th.no_grad()
def run_offline_pi_probe(
    model: Any,
    dataset_root: str | Path,
    *,
    batch_size_sequences: int = 16,
    max_seq_len: int | None = None,
    first_step_loss_weight: float = 10.0,
    logger: Any | None = None,
) -> dict[str, float]:
    if first_step_loss_weight <= 0.0:
        raise ValueError("first_step_loss_weight must be positive")
    policy: PathIntegrationRecurrentActorCriticPolicy = model.policy
    policy.set_training_mode(False)
    n_lstm_layers, lstm_hidden_size = _policy_lstm_shape(policy)

    metric_rows = []
    for batch in load_offline_pi_batches(
        dataset_root,
        batch_size_sequences=batch_size_sequences,
        max_seq_len=max_seq_len,
        shuffle=False,
        device=policy.device,
        n_lstm_layers=n_lstm_layers,
        lstm_hidden_size=lstm_hidden_size,
    ):
        _loss, metrics = compute_offline_pi_loss(
            policy,
            batch,
            first_step_loss_weight=first_step_loss_weight,
        )
        metric_rows.append(metrics)

    metrics = _mean_metrics(metric_rows, "offline_pi/probe")
    _record_metrics(logger, metrics)
    return metrics
