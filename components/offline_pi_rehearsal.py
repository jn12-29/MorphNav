from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch as th
import zarr

from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
    POINTMAZE_POLICY_OBS_KEYS,
)
from components.path_integration import soft_place_cell_cross_entropy
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


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
                sequences.append(_OfflinePISequence(obs=seq_obs, target_pos=target_pos, length=end - start))

    return sequences


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


def compute_offline_pi_loss(
    policy: PathIntegrationRecurrentActorCriticPolicy,
    batch: OfflinePIBatch,
) -> tuple[th.Tensor, dict[str, float]]:
    pi_outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
    pc_targets = policy.path_integration_target_encoder(batch.target_pos).to(dtype=pi_outputs.pc_logits.dtype)
    loss = soft_place_cell_cross_entropy(pi_outputs.pc_logits, pc_targets, mask=batch.mask)

    with th.no_grad():
        probs = th.softmax(pi_outputs.pc_logits, dim=-1)
        centers = policy.path_integration_target_encoder.centers.to(device=probs.device, dtype=probs.dtype)
        pred_pos = probs @ centers
        per_step_mse = (pred_pos - batch.target_pos.to(dtype=pred_pos.dtype)).square().mean(dim=-1)
        valid = batch.mask.bool()
        localization_mse = per_step_mse[valid].mean().item() if th.any(valid) else 0.0

    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "localization_mse": float(localization_mse),
        "masked_steps": float(batch.mask.sum().detach().cpu().item()),
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
        return {f"{prefix}/loss": float("nan"), f"{prefix}/steps": 0.0, f"{prefix}/sequence_count": 0.0}
    weighted_steps = sum(row["masked_steps"] for row in metric_rows)
    if weighted_steps > 0:
        loss = sum(row["loss"] * row["masked_steps"] for row in metric_rows) / weighted_steps
        localization_mse = sum(row["localization_mse"] * row["masked_steps"] for row in metric_rows) / weighted_steps
    else:
        loss = float(np.mean([row["loss"] for row in metric_rows]))
        localization_mse = float(np.mean([row["localization_mse"] for row in metric_rows]))
    return {
        f"{prefix}/loss": float(loss),
        f"{prefix}/localization_mse": float(localization_mse),
        f"{prefix}/steps": float(weighted_steps),
        f"{prefix}/sequence_count": float(sum(row["sequence_count"] for row in metric_rows)),
    }


def _record_metrics(logger: Any | None, metrics: dict[str, float]) -> None:
    if logger is None:
        return
    record = getattr(logger, "record", None)
    if callable(record):
        for key, value in metrics.items():
            record(key, value)


def run_offline_pi_rehearsal(
    model: Any,
    dataset_root: str | Path,
    *,
    optimizer: th.optim.Optimizer | None = None,
    lr: float = 1e-4,
    batch_size_sequences: int = 16,
    max_seq_len: int | None = None,
    max_updates: int | None = None,
    n_epochs: int = 1,
    shuffle: bool = True,
    seed: int | None = None,
    logger: Any | None = None,
) -> dict[str, float]:
    policy: PathIntegrationRecurrentActorCriticPolicy = model.policy
    policy.set_training_mode(True)
    if optimizer is policy.optimizer:
        raise ValueError("offline PI rehearsal must use an optimizer separate from policy.optimizer")
    offline_optimizer = optimizer or make_offline_pi_optimizer(policy, lr=lr)
    n_lstm_layers, lstm_hidden_size = _policy_lstm_shape(policy)

    metric_rows: list[dict[str, float]] = []
    updates = 0
    for epoch in range(n_epochs):
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
            loss, metrics = compute_offline_pi_loss(policy, batch)
            loss.backward()
            offline_optimizer.step()
            metric_rows.append(metrics)
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
        if max_updates is not None and updates >= max_updates:
            break

    metrics = _mean_metrics(metric_rows, "offline_pi")
    metrics["offline_pi/updates"] = float(updates)
    _record_metrics(logger, metrics)
    return metrics


@th.no_grad()
def run_offline_pi_probe(
    model: Any,
    dataset_root: str | Path,
    *,
    batch_size_sequences: int = 16,
    max_seq_len: int | None = None,
    logger: Any | None = None,
) -> dict[str, float]:
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
        _loss, metrics = compute_offline_pi_loss(policy, batch)
        metric_rows.append(metrics)

    metrics = _mean_metrics(metric_rows, "offline_pi/probe")
    _record_metrics(logger, metrics)
    return metrics
