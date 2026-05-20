from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def make_run_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "tensorboard": output_dir / "tensorboard",
        "models": output_dir / "models",
        "eval": output_dir / "eval",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_train_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"offline_pi.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def dataset_metadata_summary(dataset_root: Path | None) -> dict[str, Any] | None:
    if dataset_root is None:
        return None
    summary: dict[str, Any] = {"root": str(dataset_root)}
    metadata_path = dataset_root / "dataset_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary["metadata_error"] = str(exc)
        else:
            summary["metadata"] = metadata
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary["manifest_error"] = str(exc)
        else:
            episodes = manifest.get("episodes")
            shards = manifest.get("shards")
            summary["manifest"] = {
                key: value
                for key, value in manifest.items()
                if key not in {"episodes", "shards"}
            }
            if isinstance(episodes, list):
                summary["manifest"]["episode_count"] = len(episodes)
            if isinstance(shards, list):
                summary["manifest"]["shard_count"] = len(shards)
    return summary


def make_event(
    event: str,
    *,
    phase: str,
    epoch: int,
    update: int,
    global_step: int,
    metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "phase": phase,
        "epoch": int(epoch),
        "update": int(update),
        "global_step": int(global_step),
        "timestamp": utc_timestamp(),
    }
    if metrics:
        payload.update(metrics)
    payload.update(extra)
    return payload


class TensorBoardRunWriter:
    def __init__(self, log_dir: Path | None, *, enabled: bool) -> None:
        self.log_dir = log_dir
        self.writer = None
        self.disabled_reason: str | None = None
        if not enabled:
            self.disabled_reason = "disabled by CLI"
            return
        if log_dir is None:
            raise ValueError("TensorBoard log_dir is required when TensorBoard is enabled")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            self.disabled_reason = (
                "TensorBoard dependencies are unavailable; continuing with JSON and text logging only "
                f"({exc})"
            )
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is not None and hasattr(self.writer, "add_text"):
            self.writer.add_text(tag, text, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def tensorboard_log_update(writer: TensorBoardRunWriter, metrics: dict[str, float], *, lr: float, step: int) -> None:
    writer.add_scalar("train/loss_step", float(metrics["loss"]), step)
    writer.add_scalar("train/localization_mse_step", float(metrics["localization_mse"]), step)
    writer.add_scalar("train/lr", float(lr), step)


def tensorboard_log_epoch(writer: TensorBoardRunWriter, metrics: dict[str, float], *, step: int) -> None:
    writer.add_scalar("train/loss_epoch", float(metrics["offline_pi/loss"]), step)
    writer.add_scalar("train/localization_mse_epoch", float(metrics["offline_pi/localization_mse"]), step)
    writer.add_scalar("train/epoch_seconds", float(metrics["offline_pi/epoch_seconds"]), step)


def tensorboard_log_probe(writer: TensorBoardRunWriter, metrics: dict[str, float], *, step: int) -> None:
    mapping = {
        "probe/loss": "offline_pi/probe/loss",
        "probe/localization_mse": "offline_pi/probe/localization_mse",
        "probe/localization_rmse": "offline_pi/probe/localization_rmse",
        "probe/localization_mae": "offline_pi/probe/localization_mae",
        "probe/x_mae": "offline_pi/probe/x_mae",
        "probe/y_mae": "offline_pi/probe/y_mae",
    }
    for tag, key in mapping.items():
        writer.add_scalar(tag, float(metrics[key]), step)


def save_model_checkpoint(
    model: Any,
    model_path: Path,
    *,
    epoch: int,
    update: int,
    global_step: int,
    config: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    probe_dataset_info: dict[str, Any] | None,
    latest_train_metrics: dict[str, float] | None,
    latest_probe_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_model_path = model_path.with_name(f".{model_path.stem}.tmp.zip")
    model.save(tmp_model_path)
    os.replace(tmp_model_path, model_path)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "epoch": int(epoch),
        "update": int(update),
        "global_step": int(global_step),
        "config": config,
        "dataset_info": dataset_info,
        "probe_dataset_info": probe_dataset_info,
        "latest_train_metrics": latest_train_metrics,
        "latest_probe_metrics": latest_probe_metrics,
        "model_path": str(model_path),
        "created_at": utc_timestamp(),
        "future_resume_state": {},
    }
    write_json_atomic(model_path.with_suffix(".json"), metadata)
    return metadata
