from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphnav_matplotlib")

import gymnasium as gym
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import CustomCombinedExtractor, PathIntegrationRecurrentPPO
from components.dataset_gen.pointmaze_config import make_phase1_pointmaze_pi_env_config
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs
from components.offline_pi_workflow import run_offline_pi_workflow

_ZOO_EVAL_GLOBALS = {"CustomCombinedExtractor": CustomCombinedExtractor}

PI_ZOO_CONFIG_PATH = REPO_ROOT / "rl-baselines3-zoo" / "conf" / "maze_pi.yml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone offline PI rehearsal/probe for PointMaze.")
    parser.add_argument("--mode", choices=("train", "probe"), default="train")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--probe-dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size-sequences", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every-updates", type=int, default=0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--eval-at-start", dest="eval_at_start", action="store_true", default=True)
    parser.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.add_argument("--eval-artifact-every-epochs", type=int, default=0)
    parser.add_argument("--eval-gridscore-every-epochs", type=int, default=0)
    parser.add_argument("--gridscore-n-bins", type=int, default=32)
    parser.add_argument("--gridscore-max-steps", type=int, default=None)
    parser.add_argument("--gridscore-top-k", type=int, default=8)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=0)
    parser.add_argument("--save-final-checkpoint", dest="save_final_checkpoint", action="store_true", default=True)
    parser.add_argument("--no-save-final-checkpoint", dest="save_final_checkpoint", action="store_false")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--tensorboard-log-dir", type=Path, default=None)
    return parser


def resolve_output_dir(output_dir: Path | None, run_name: str | None, seed: int) -> Path:
    if output_dir is not None:
        return output_dir
    resolved_run_name = run_name or f"pointmaze_phase1_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Path("runs/offline_pi") / resolved_run_name


def resolved_run_name(output_dir: Path, run_name: str | None) -> str:
    return run_name or output_dir.name


def _make_env():
    env_config = make_phase1_pointmaze_pi_env_config()
    return gym.make("PointMaze", **build_pointmaze_env_kwargs(env_config))


def load_pointmaze_pi_hyperparams(config_path: Path = PI_ZOO_CONFIG_PATH) -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return deepcopy(config["PointMaze"])


def _eval_zoo_value(value):
    if isinstance(value, str):
        return eval(value, _ZOO_EVAL_GLOBALS)
    return value


def fresh_model_kwargs(*, learning_rate: float, seed: int, device: str, config_path: Path = PI_ZOO_CONFIG_PATH) -> dict:
    hyperparams = load_pointmaze_pi_hyperparams(config_path)
    policy = hyperparams.pop("policy")
    hyperparams.pop("n_envs", None)
    hyperparams.pop("n_timesteps", None)
    hyperparams["learning_rate"] = learning_rate
    hyperparams["seed"] = seed
    hyperparams["device"] = device
    hyperparams["verbose"] = 0
    if "policy_kwargs" in hyperparams:
        hyperparams["policy_kwargs"] = _eval_zoo_value(hyperparams["policy_kwargs"])
    return {"policy": policy, "kwargs": hyperparams}


def _make_fresh_model(env, *, learning_rate: float, seed: int, device: str) -> PathIntegrationRecurrentPPO:
    model_config = fresh_model_kwargs(learning_rate=learning_rate, seed=seed, device=device)
    return PathIntegrationRecurrentPPO(
        model_config["policy"],
        env,
        **model_config["kwargs"],
    )


def _load_or_create_model(args: argparse.Namespace, env) -> PathIntegrationRecurrentPPO:
    if args.model_path is not None:
        return PathIntegrationRecurrentPPO.load(args.model_path, env=env, device=args.device)
    if args.mode == "probe":
        raise ValueError("--model-path is required in probe mode")
    return _make_fresh_model(env, learning_rate=args.learning_rate, seed=args.seed, device=args.device)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size_sequences <= 0:
        parser.error("--batch-size-sequences must be > 0")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        parser.error("--max-seq-len must be > 0")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.max_updates is not None and args.max_updates <= 0:
        parser.error("--max-updates must be > 0 when provided")
    if args.log_every_updates < 0:
        parser.error("--log-every-updates must be >= 0")
    if args.eval_every_epochs < 0:
        parser.error("--eval-every-epochs must be >= 0")
    if args.eval_artifact_every_epochs < 0:
        parser.error("--eval-artifact-every-epochs must be >= 0")
    if args.eval_gridscore_every_epochs < 0:
        parser.error("--eval-gridscore-every-epochs must be >= 0")
    if args.gridscore_n_bins <= 0:
        parser.error("--gridscore-n-bins must be > 0")
    if args.gridscore_max_steps is not None and args.gridscore_max_steps <= 0:
        parser.error("--gridscore-max-steps must be > 0 when provided")
    if args.gridscore_top_k <= 0:
        parser.error("--gridscore-top-k must be > 0")
    if args.checkpoint_every_epochs < 0:
        parser.error("--checkpoint-every-epochs must be >= 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    output_dir = resolve_output_dir(args.output_dir, args.run_name, args.seed)
    run_name = resolved_run_name(output_dir, args.run_name)
    fresh_settings = fresh_model_kwargs(learning_rate=args.learning_rate, seed=args.seed, device=args.device)
    env = _make_env()
    try:
        model = _load_or_create_model(args, env)
        metrics = run_offline_pi_workflow(
            model,
            args,
            output_dir=output_dir,
            run_name=run_name,
            fresh_model_settings=fresh_settings,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
