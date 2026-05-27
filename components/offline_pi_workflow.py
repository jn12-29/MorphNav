from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from components.offline_pi_rehearsal import count_offline_pi_sequences, run_offline_pi_probe, run_offline_pi_rehearsal
from components.offline_pi_runtime import (
    TensorBoardRunWriter,
    append_jsonl,
    configure_train_logger,
    dataset_metadata_summary,
    make_event,
    make_run_dirs,
    save_model_checkpoint,
    tensorboard_log_epoch,
    tensorboard_log_probe,
    tensorboard_log_update,
    utc_timestamp,
    write_json_atomic,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def _resolved_tensorboard_log_dir(args: Any, dirs: dict[str, Path]) -> Path | None:
    if args.tensorboard_log_dir is not None:
        return args.tensorboard_log_dir
    if args.tensorboard:
        return dirs["tensorboard"]
    return None


def _effective_config(
    args: Any,
    *,
    output_dir: Path,
    run_name: str,
    tensorboard_log_dir: Path | None,
    fresh_model_settings: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    probe_dataset_info: dict[str, Any] | None,
) -> dict[str, Any]:
    cli_args = {key: _jsonable(value) for key, value in vars(args).items()}
    return {
        "created_at": utc_timestamp(),
        "mode": args.mode,
        "cli_args": cli_args,
        "output_dir": str(output_dir),
        "run_name": run_name,
        "dataset_root": str(args.dataset_root),
        "probe_dataset_root": str(args.probe_dataset_root) if args.probe_dataset_root is not None else None,
        "learning_rate": args.learning_rate,
        "batch_size_sequences": args.batch_size_sequences,
        "max_seq_len": args.max_seq_len,
        "max_updates": args.max_updates,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
        "log_every_updates": args.log_every_updates,
        "eval_every_epochs": args.eval_every_epochs,
        "eval_at_start": args.eval_at_start,
        "tensorboard_requested": args.tensorboard,
        "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else None,
        "eval_artifact_every_epochs": args.eval_artifact_every_epochs,
        "eval_gridscore_every_epochs": args.eval_gridscore_every_epochs,
        "gridscore_n_bins": args.gridscore_n_bins,
        "gridscore_max_steps": args.gridscore_max_steps,
        "gridscore_top_k": args.gridscore_top_k,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "save_final_checkpoint": args.save_final_checkpoint,
        "fresh_model_settings": _jsonable(fresh_model_settings),
        "dataset_info": dataset_info,
        "probe_dataset_info": probe_dataset_info,
    }


def _bounded_log_every(args: Any) -> int:
    if args.log_every_updates > 0:
        return args.log_every_updates
    try:
        sequence_count = count_offline_pi_sequences(args.dataset_root, max_seq_len=args.max_seq_len)
    except (FileNotFoundError, ValueError):
        return 1
    updates_per_epoch = max(1, (sequence_count + args.batch_size_sequences - 1) // args.batch_size_sequences)
    if args.max_updates is not None:
        updates_per_epoch = min(updates_per_epoch, args.max_updates)
    return max(1, updates_per_epoch // 10)


def _probe_epoch_path(metrics_dir: Path, epoch: int) -> Path:
    return metrics_dir / f"probe_epoch_{epoch:04d}.json"


def _prefix_update_metrics(metrics: dict[str, float], *, lr: float) -> dict[str, float]:
    return {
        "offline_pi/loss_step": float(metrics["loss"]),
        "offline_pi/localization_mse_step": float(metrics["localization_mse"]),
        "offline_pi/first_localization_mse_step": float(metrics["first_localization_mse"]),
        "offline_pi/first_localization_mse_ratio_step": float(metrics["first_localization_mse_ratio"]),
        "offline_pi/masked_steps": float(metrics["masked_steps"]),
        "offline_pi/first_step_count": float(metrics["first_step_count"]),
        "offline_pi/sequence_count": float(metrics["sequence_count"]),
        "offline_pi/lr": float(lr),
    }


def _event_epoch_metrics(metrics: dict[str, float], *, samples_seen: int) -> dict[str, float]:
    return {
        "offline_pi/loss_mean": float(metrics["offline_pi/loss"]),
        "offline_pi/loss_std": float(metrics["offline_pi/loss_std"]),
        "offline_pi/localization_mse_mean": float(metrics["offline_pi/localization_mse"]),
        "offline_pi/localization_mse_std": float(metrics["offline_pi/localization_mse_std"]),
        "offline_pi/first_localization_mse_mean": float(metrics["offline_pi/first_localization_mse"]),
        "offline_pi/first_localization_mse_std": float(metrics["offline_pi/first_localization_mse_std"]),
        "offline_pi/first_localization_mse_ratio": float(metrics["offline_pi/first_localization_mse_ratio"]),
        "offline_pi/steps": float(metrics["offline_pi/steps"]),
        "offline_pi/first_step_count": float(metrics["offline_pi/first_step_count"]),
        "offline_pi/sequence_count": float(metrics["offline_pi/sequence_count"]),
        "offline_pi/updates": float(metrics["offline_pi/updates"]),
        "offline_pi/epoch_seconds": float(metrics["offline_pi/epoch_seconds"]),
        "offline_pi/samples_seen": float(samples_seen),
    }


def _final_epoch_from_metrics(metrics: dict[str, float], fallback: int) -> int:
    return int(metrics.get("offline_pi/final_epoch", float(fallback)))


def _should_export_gridscore(args: Any, epoch: int) -> bool:
    return args.eval_gridscore_every_epochs > 0 and epoch % args.eval_gridscore_every_epochs == 0


def _gridscore_output_dir(eval_dir: Path, epoch: int) -> Path:
    return eval_dir / f"gridscore_epoch_{epoch:04d}"


def _run_probe(
    model: Any,
    dataset_root: Path,
    *,
    epoch: int,
    update: int,
    global_step: int,
    args: Any,
    dirs: dict[str, Path],
    logger: logging.Logger,
    metrics_jsonl: Path,
    tensorboard_writer: TensorBoardRunWriter,
    export_artifacts: bool,
    export_gridscore: bool,
) -> dict[str, float]:
    logger.info("probe start epoch=%04d dataset=%s", epoch, dataset_root)
    started = time.time()
    metrics = run_offline_pi_probe(
        model,
        dataset_root,
        batch_size_sequences=args.batch_size_sequences,
        max_seq_len=args.max_seq_len,
    )
    metrics["offline_pi/probe/seconds"] = float(time.time() - started)
    artifact_summary = None
    if export_artifacts:
        from components.offline_pi_eval_artifacts import export_probe_artifacts

        artifact_summary = export_probe_artifacts(
            model,
            dataset_root,
            dirs["eval"],
            epoch=epoch,
            batch_size_sequences=args.batch_size_sequences,
            max_seq_len=args.max_seq_len,
        )
    gridscore_summary = None
    if export_gridscore:
        from components.offline_pi_gridscore import export_gridscore_artifacts, gridscore_metrics_from_summary

        gridscore_started = time.time()
        gridscore_summary = export_gridscore_artifacts(
            model,
            dataset_root,
            _gridscore_output_dir(dirs["eval"], epoch),
            batch_size_sequences=args.batch_size_sequences,
            max_seq_len=args.max_seq_len,
            n_bins=args.gridscore_n_bins,
            max_steps=args.gridscore_max_steps,
            max_units=None,
            top_k=args.gridscore_top_k,
        )
        metrics.update(gridscore_metrics_from_summary(gridscore_summary, seconds=time.time() - gridscore_started))
    write_json_atomic(_probe_epoch_path(dirs["metrics"], epoch), metrics)
    append_jsonl(
        metrics_jsonl,
        make_event(
            "probe",
            phase="probe",
            epoch=epoch,
            update=update,
            global_step=global_step,
            metrics=metrics,
            artifact_summary=artifact_summary,
            gridscore_summary=gridscore_summary,
        ),
    )
    tensorboard_log_probe(tensorboard_writer, metrics, step=global_step)
    logger.info(
        "probe complete epoch=%04d loss=%.6f localization_mse=%.6f first_localization_mse=%.6f first_ratio=%.4f rmse=%.6f",
        epoch,
        metrics["offline_pi/probe/loss"],
        metrics["offline_pi/probe/localization_mse"],
        metrics["offline_pi/probe/first_localization_mse"],
        metrics["offline_pi/probe/first_localization_mse_ratio"],
        metrics["offline_pi/probe/localization_rmse"],
    )
    if gridscore_summary is not None:
        logger.info(
            "gridscore complete epoch=%04d best=%.6f best_unit=%d valid_units=%d",
            epoch,
            gridscore_summary["best_grid_score"],
            gridscore_summary["best_unit"],
            gridscore_summary["valid_units"],
        )
    return metrics


def _save_checkpoint(
    model: Any,
    path: Path,
    *,
    epoch: int,
    update: int,
    global_step: int,
    config: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    probe_dataset_info: dict[str, Any] | None,
    latest_train_metrics: dict[str, float] | None,
    latest_probe_metrics: dict[str, float] | None,
    logger: logging.Logger,
    metrics_jsonl: Path,
) -> None:
    metadata = save_model_checkpoint(
        model,
        path,
        epoch=epoch,
        update=update,
        global_step=global_step,
        config=config,
        dataset_info=dataset_info,
        probe_dataset_info=probe_dataset_info,
        latest_train_metrics=latest_train_metrics,
        latest_probe_metrics=latest_probe_metrics,
    )
    append_jsonl(
        metrics_jsonl,
        make_event(
            "checkpoint",
            phase="checkpoint",
            epoch=epoch,
            update=update,
            global_step=global_step,
            model_path=str(path),
            metadata_path=str(path.with_suffix(".json")),
        ),
    )
    logger.info("checkpoint written epoch=%04d path=%s", epoch, metadata["model_path"])


def _run_train(
    model: Any,
    args: Any,
    *,
    dirs: dict[str, Path],
    config: dict[str, Any],
    dataset_info: dict[str, Any] | None,
    probe_dataset_info: dict[str, Any] | None,
    logger: logging.Logger,
    metrics_jsonl: Path,
    tensorboard_writer: TensorBoardRunWriter,
) -> dict[str, float]:
    log_every = _bounded_log_every(args)
    latest_probe_metrics: dict[str, float] | None = None
    latest_epoch_metrics: dict[str, float] | None = None
    update_state = {"update": 0, "global_step": 0}

    if args.probe_dataset_root is not None and args.eval_at_start:
        latest_probe_metrics = _run_probe(
            model,
            args.probe_dataset_root,
            epoch=0,
            update=0,
            global_step=0,
            args=args,
            dirs=dirs,
            logger=logger,
            metrics_jsonl=metrics_jsonl,
            tensorboard_writer=tensorboard_writer,
            export_artifacts=args.eval_artifact_every_epochs > 0,
            export_gridscore=_should_export_gridscore(args, 0),
        )

    def on_epoch_start(payload: dict[str, Any]) -> None:
        logger.info("epoch start epoch=%04d update=%d", payload["epoch"], payload["update"])

    def on_update(payload: dict[str, Any]) -> None:
        update_state["update"] = int(payload["update"])
        update_state["global_step"] += int(payload["metrics"]["masked_steps"])
        if payload["update"] % log_every == 0:
            metrics = _prefix_update_metrics(payload["metrics"], lr=payload["lr"])
            append_jsonl(
                metrics_jsonl,
                make_event(
                    "train_update",
                    phase="train",
                    epoch=payload["epoch"],
                    update=payload["update"],
                    global_step=update_state["global_step"],
                    metrics=metrics,
                ),
            )
            tensorboard_log_update(
                tensorboard_writer,
                payload["metrics"],
                lr=payload["lr"],
                step=update_state["global_step"],
            )
            logger.info(
                "update progress epoch=%04d update=%d loss=%.6f localization_mse=%.6f first_localization_mse=%.6f masked_steps=%d",
                payload["epoch"],
                payload["update"],
                payload["metrics"]["loss"],
                payload["metrics"]["localization_mse"],
                payload["metrics"]["first_localization_mse"],
                int(payload["metrics"]["masked_steps"]),
            )

    def on_epoch_end(payload: dict[str, Any]) -> None:
        nonlocal latest_epoch_metrics, latest_probe_metrics
        latest_epoch_metrics = payload["metrics"]
        append_jsonl(
            metrics_jsonl,
            make_event(
                "train_epoch",
                phase="train",
                epoch=payload["epoch"],
                update=payload["update"],
                global_step=update_state["global_step"],
                metrics=_event_epoch_metrics(payload["metrics"], samples_seen=update_state["global_step"]),
            ),
        )
        tensorboard_log_epoch(tensorboard_writer, payload["metrics"], step=update_state["global_step"])
        logger.info(
            "epoch complete epoch=%04d updates=%d loss=%.6f localization_mse=%.6f first_localization_mse=%.6f first_ratio=%.4f seconds=%.2f",
            payload["epoch"],
            int(payload["metrics"]["offline_pi/updates"]),
            payload["metrics"]["offline_pi/loss"],
            payload["metrics"]["offline_pi/localization_mse"],
            payload["metrics"]["offline_pi/first_localization_mse"],
            payload["metrics"]["offline_pi/first_localization_mse_ratio"],
            payload["metrics"]["offline_pi/epoch_seconds"],
        )
        if args.probe_dataset_root is not None and args.eval_every_epochs > 0 and payload["epoch"] % args.eval_every_epochs == 0:
            latest_probe_metrics = _run_probe(
                model,
                args.probe_dataset_root,
                epoch=payload["epoch"],
                update=payload["update"],
                global_step=update_state["global_step"],
                args=args,
                dirs=dirs,
                logger=logger,
                metrics_jsonl=metrics_jsonl,
                tensorboard_writer=tensorboard_writer,
                export_artifacts=(
                    args.eval_artifact_every_epochs > 0
                    and payload["epoch"] % args.eval_artifact_every_epochs == 0
                ),
                export_gridscore=_should_export_gridscore(args, payload["epoch"]),
            )
        if args.checkpoint_every_epochs > 0 and payload["epoch"] % args.checkpoint_every_epochs == 0:
            _save_checkpoint(
                model,
                dirs["models"] / f"checkpoint_epoch_{payload['epoch']:04d}.zip",
                epoch=payload["epoch"],
                update=payload["update"],
                global_step=update_state["global_step"],
                config=config,
                dataset_info=dataset_info,
                probe_dataset_info=probe_dataset_info,
                latest_train_metrics=latest_epoch_metrics,
                latest_probe_metrics=latest_probe_metrics,
                logger=logger,
                metrics_jsonl=metrics_jsonl,
            )

    train_metrics = run_offline_pi_rehearsal(
        model,
        args.dataset_root,
        lr=args.learning_rate,
        batch_size_sequences=args.batch_size_sequences,
        max_seq_len=args.max_seq_len,
        max_updates=args.max_updates,
        n_epochs=args.epochs,
        seed=args.seed,
        on_epoch_start=on_epoch_start,
        on_update=on_update,
        on_epoch_end=on_epoch_end,
    )
    metrics = dict(train_metrics)
    if latest_probe_metrics is not None:
        metrics.update(latest_probe_metrics)

    if args.save_final_checkpoint:
        final_epoch = _final_epoch_from_metrics(train_metrics, args.epochs)
        _save_checkpoint(
            model,
            dirs["models"] / "final_model.zip",
            epoch=final_epoch,
            update=update_state["update"],
            global_step=update_state["global_step"],
            config=config,
            dataset_info=dataset_info,
            probe_dataset_info=probe_dataset_info,
            latest_train_metrics=latest_epoch_metrics or train_metrics,
            latest_probe_metrics=latest_probe_metrics,
            logger=logger,
            metrics_jsonl=metrics_jsonl,
        )
        logger.info("final model written path=%s", dirs["models"] / "final_model.zip")
    return metrics


def _run_probe_only(
    model: Any,
    args: Any,
    *,
    dirs: dict[str, Path],
    logger: logging.Logger,
    metrics_jsonl: Path,
    tensorboard_writer: TensorBoardRunWriter,
) -> dict[str, float]:
    return _run_probe(
        model,
        args.dataset_root,
        epoch=0,
        update=0,
        global_step=0,
        args=args,
        dirs=dirs,
        logger=logger,
        metrics_jsonl=metrics_jsonl,
        tensorboard_writer=tensorboard_writer,
        export_artifacts=args.eval_artifact_every_epochs > 0,
        export_gridscore=_should_export_gridscore(args, 0),
    )


def run_offline_pi_workflow(
    model: Any,
    args: Any,
    *,
    output_dir: Path,
    run_name: str,
    fresh_model_settings: dict[str, Any],
) -> dict[str, float]:
    dirs = make_run_dirs(output_dir)
    logger = configure_train_logger(output_dir)
    metrics_jsonl = dirs["metrics"] / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    dataset_info = dataset_metadata_summary(args.dataset_root)
    probe_dataset_info = dataset_metadata_summary(args.probe_dataset_root)
    tensorboard_log_dir = _resolved_tensorboard_log_dir(args, dirs)
    config = _effective_config(
        args,
        output_dir=output_dir,
        run_name=run_name,
        tensorboard_log_dir=tensorboard_log_dir,
        fresh_model_settings=fresh_model_settings,
        dataset_info=dataset_info,
        probe_dataset_info=probe_dataset_info,
    )
    write_json_atomic(output_dir / "config.json", config)

    tensorboard_writer = TensorBoardRunWriter(tensorboard_log_dir, enabled=args.tensorboard)
    config["tensorboard_enabled"] = tensorboard_writer.enabled
    config["tensorboard_disabled_reason"] = tensorboard_writer.disabled_reason
    write_json_atomic(output_dir / "config.json", config)
    try:
        tensorboard_writer.add_text("run/config", json.dumps(config, indent=2, sort_keys=True))
        if tensorboard_writer.enabled:
            logger.info("tensorboard enabled log_dir=%s", tensorboard_log_dir)
        elif tensorboard_writer.disabled_reason != "disabled by CLI":
            logger.info("tensorboard disabled: %s", tensorboard_writer.disabled_reason)
        logger.info(
            "run start output_dir=%s dataset_root=%s probe_dataset_root=%s device=%s seed=%d epochs=%d max_seq_len=%s batch_size_sequences=%d",
            output_dir,
            args.dataset_root,
            args.probe_dataset_root,
            args.device,
            args.seed,
            args.epochs,
            args.max_seq_len,
            args.batch_size_sequences,
        )
        logger.info("config written path=%s", output_dir / "config.json")
        append_jsonl(
            metrics_jsonl,
            make_event(
                "run_start",
                phase=args.mode,
                epoch=0,
                update=0,
                global_step=0,
                output_dir=str(output_dir),
                dataset_root=str(args.dataset_root),
                probe_dataset_root=str(args.probe_dataset_root) if args.probe_dataset_root is not None else None,
            ),
        )

        if args.mode == "train":
            metrics = _run_train(
                model,
                args,
                dirs=dirs,
                config=config,
                dataset_info=dataset_info,
                probe_dataset_info=probe_dataset_info,
                logger=logger,
                metrics_jsonl=metrics_jsonl,
                tensorboard_writer=tensorboard_writer,
            )
        else:
            metrics = _run_probe_only(
                model,
                args,
                dirs=dirs,
                logger=logger,
                metrics_jsonl=metrics_jsonl,
                tensorboard_writer=tensorboard_writer,
            )

        write_json_atomic(dirs["metrics"] / "offline_pi_metrics.json", metrics)
        append_jsonl(
            metrics_jsonl,
            make_event(
                "run_end",
                phase=args.mode,
                epoch=_final_epoch_from_metrics(metrics, args.epochs) if args.mode == "train" else 0,
                update=int(metrics.get("offline_pi/updates", 0.0)),
                global_step=int(metrics.get("offline_pi/steps", metrics.get("offline_pi/probe/steps", 0.0))),
                metrics=metrics,
            ),
        )
        logger.info("final metrics written path=%s", dirs["metrics"] / "offline_pi_metrics.json")
        return metrics
    finally:
        tensorboard_writer.close()
