from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import PathIntegrationRecurrentPPO
from components.pointmaze_trajectory_rendering import (
    collect_recurrent_policy_rollout,
    load_dataset_episode,
    predict_dataset_episode_pi,
    resolve_pointmaze_env_kwargs,
    resolve_render_output_dir,
    write_render_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PointMaze Phase 1 trajectories as 3D + top-down MP4 artifacts.")
    parser.add_argument("--mode", choices=("dataset", "probe", "rollout"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"))
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode in {"dataset", "probe"}:
        if args.dataset_root is None:
            parser.error(f"--dataset-root is required in {args.mode} mode")
        if not args.episodes:
            parser.error(f"--episodes must contain at least one episode id in {args.mode} mode")
    if args.mode in {"probe", "rollout"} and args.model_path is None:
        parser.error(f"--model-path is required in {args.mode} mode")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.stride <= 0:
        parser.error("--stride must be > 0")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be > 0")
    if args.steps <= 0:
        parser.error("--steps must be > 0")
    if args.width <= 1 or args.height <= 0:
        parser.error("--width must be > 1 and --height must be > 0")
    if args.bounds is not None and (args.bounds[1] <= args.bounds[0] or args.bounds[3] <= args.bounds[2]):
        parser.error("--bounds must satisfy MIN_X < MAX_X and MIN_Y < MAX_Y")


def _command_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "dataset_root": None if args.dataset_root is None else str(args.dataset_root),
        "episodes": args.episodes,
        "model_path": None if args.model_path is None else str(args.model_path),
        "output_dir": None if args.output_dir is None else str(args.output_dir),
        "fps": args.fps,
        "stride": args.stride,
        "max_seq_len": args.max_seq_len,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "deterministic": args.deterministic,
        "width": args.width,
        "height": args.height,
        "bounds": args.bounds,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def _make_render_env(env_kwargs: dict[str, Any]):
    kwargs = dict(env_kwargs)
    kwargs["render_mode"] = "rgb_array"
    return gym.make("PointMaze", **kwargs)


def _load_model(model_path: Path, env, device: str) -> PathIntegrationRecurrentPPO:
    return PathIntegrationRecurrentPPO.load(model_path, env=env, device=device)


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_dataset_or_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    env_resolution = resolve_pointmaze_env_kwargs(args.dataset_root)
    output_dir = resolve_render_output_dir(
        mode=args.mode,
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _make_render_env(env_resolution.env_kwargs)
    try:
        model = _load_model(args.model_path, env, args.device) if args.mode == "probe" else None
        summaries: list[dict[str, Any]] = []
        for episode_id in args.episodes:
            trajectory = load_dataset_episode(args.dataset_root, episode_id, kind=args.mode)
            if model is not None:
                trajectory = predict_dataset_episode_pi(
                    model,
                    trajectory,
                    max_seq_len=args.max_seq_len,
                )
            output_base = output_dir / f"episode_{episode_id:06d}"
            summary = write_render_artifacts(
                trajectory,
                env,
                output_base,
                fps=args.fps,
                stride=args.stride,
                bounds=None if args.bounds is None else tuple(float(value) for value in args.bounds),
                output_size=(args.width, args.height),
                command_config=_command_config(args),
                env_kwargs=env_resolution.env_kwargs,
                env_substitutions=env_resolution.substitutions,
                dataset_root=args.dataset_root,
                model_path=args.model_path,
            )
            summaries.append(summary)
            _print_summary(summary)
        return summaries
    finally:
        env.close()


def _run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    env_resolution = resolve_pointmaze_env_kwargs(args.dataset_root)
    output_dir = resolve_render_output_dir(
        mode="rollout",
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _make_render_env(env_resolution.env_kwargs)
    try:
        model = _load_model(args.model_path, env, args.device)
        trajectory = collect_recurrent_policy_rollout(
            model,
            env,
            steps=args.steps,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        output_base = output_dir / f"rollout_seed{args.seed}_steps{args.steps}"
        summary = write_render_artifacts(
            trajectory,
            env,
            output_base,
            fps=args.fps,
            stride=args.stride,
            bounds=None if args.bounds is None else tuple(float(value) for value in args.bounds),
            output_size=(args.width, args.height),
            command_config=_command_config(args),
            env_kwargs=env_resolution.env_kwargs,
            env_substitutions=env_resolution.substitutions,
            dataset_root=args.dataset_root,
            model_path=args.model_path,
        )
        _print_summary(summary)
        return summary
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.mode in {"dataset", "probe"}:
        _run_dataset_or_probe(args)
    else:
        _run_rollout(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
