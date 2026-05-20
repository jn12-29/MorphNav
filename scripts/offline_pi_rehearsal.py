from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import CustomCombinedExtractor, PathIntegrationRecurrentPPO
from components.dataset_gen.pointmaze_config import make_phase1_pointmaze_pi_env_config
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs
from components.offline_pi_rehearsal import run_offline_pi_probe, run_offline_pi_rehearsal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone offline PI rehearsal/probe for PointMaze.")
    parser.add_argument("--mode", choices=("train", "probe"), default="train")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--probe-dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/offline_pi/pointmaze_phase1"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size-sequences", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _make_env():
    env_config = make_phase1_pointmaze_pi_env_config()
    return gym.make("PointMaze", **build_pointmaze_env_kwargs(env_config))


def _make_fresh_model(env, *, learning_rate: float, seed: int, device: str) -> PathIntegrationRecurrentPPO:
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(drop_keys=["achieved_goal"]),
        n_lstm_layers=1,
        lstm_hidden_size=256,
        pi_bottleneck_dim=256,
        pi_dropout_rate=0.5,
    )
    return PathIntegrationRecurrentPPO(
        "PathIntegrationMultiInputLstmPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        pi_loss_coef=1.0,
        pi_target_key="achieved_goal",
        pi_n_place_cells=256,
        pi_place_cell_scale=0.01,
        pi_pos_min=-2.5,
        pi_pos_max=2.5,
        pi_neurons_seed=8341,
        seed=seed,
        device=device,
        verbose=0,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            model.save(args.output_dir / "final_model")
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

        metrics_path = args.output_dir / "offline_pi_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
