from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

import gymnasium as gym
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import CustomCombinedExtractor, PathIntegrationRecurrentPPO
from components.dataset_gen.pointmaze_config import make_phase1_pointmaze_pi_env_config
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs
from components.offline_pi_rehearsal import run_offline_pi_probe, run_offline_pi_rehearsal

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
    return parser


def resolve_output_dir(output_dir: Path | None, run_name: str | None, seed: int) -> Path:
    if output_dir is not None:
        return output_dir
    resolved_run_name = run_name or f"pointmaze_phase1_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Path("runs/offline_pi") / resolved_run_name


def _make_env():
    env_config = make_phase1_pointmaze_pi_env_config()
    return gym.make("PointMaze", **build_pointmaze_env_kwargs(env_config))


def load_pointmaze_pi_hyperparams(config_path: Path = PI_ZOO_CONFIG_PATH) -> dict:
    with Path(config_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return deepcopy(config["PointMaze"])


def _eval_zoo_value(value):
    if isinstance(value, str):
        return eval(value)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size_sequences <= 0:
        parser.error("--batch-size-sequences must be > 0")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        parser.error("--max-seq-len must be > 0")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")

    output_dir = resolve_output_dir(args.output_dir, args.run_name, args.seed)
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env()
    try:
        model = _load_or_create_model(args, env)
        metrics: dict[str, float] = {}
        if args.mode == "train":
            metrics.update(
                run_offline_pi_rehearsal(
                    model,
                    args.dataset_root,
                    lr=args.learning_rate,
                    batch_size_sequences=args.batch_size_sequences,
                    max_seq_len=args.max_seq_len,
                    max_updates=args.max_updates,
                    n_epochs=args.epochs,
                    seed=args.seed,
                )
            )
            model.save(models_dir / "final_model")
            if args.probe_dataset_root is not None:
                metrics.update(
                    run_offline_pi_probe(
                        model,
                        args.probe_dataset_root,
                        batch_size_sequences=args.batch_size_sequences,
                        max_seq_len=args.max_seq_len,
                    )
                )
        else:
            metrics.update(
                run_offline_pi_probe(
                    model,
                    args.dataset_root,
                    batch_size_sequences=args.batch_size_sequences,
                    max_seq_len=args.max_seq_len,
                )
            )

        metrics_path = metrics_dir / "offline_pi_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote offline PI outputs to {output_dir}")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
