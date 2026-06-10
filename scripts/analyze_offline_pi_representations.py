from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/morphnav_matplotlib")

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs  # noqa: F401  # Registers local Gymnasium envs.
from components import PathIntegrationRecurrentPPO
from components.offline_pi_gridscore import (
    compute_gridscore_analysis,
    save_gridscore_analysis,
)
from components.dataset_gen.pointmaze_config import make_phase1_pointmaze_pi_env_config
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze offline PI bottleneck spatial representations.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size-sequences", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--gridscore-positive-activations", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def resolve_output_dir(model_path: Path, dataset_root: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    run_root = model_path.parent.parent if model_path.parent.name == "models" else model_path.parent
    return run_root / "analysis" / "representations" / dataset_root.name


def write_analysis_config(args: argparse.Namespace, output_dir: Path, bounds: tuple[float, float, float, float]) -> None:
    payload = {
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "output_dir": str(output_dir),
        "batch_size_sequences": args.batch_size_sequences,
        "max_seq_len": args.max_seq_len,
        "n_bins": args.n_bins,
        "bounds": [float(value) for value in bounds],
        "max_steps": args.max_steps,
        "max_units": args.max_units,
        "top_k": args.top_k,
        "gridscore_positive_activations": bool(getattr(args, "gridscore_positive_activations", False)),
        "device": args.device,
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_env():
    env_config = make_phase1_pointmaze_pi_env_config()
    return gym.make("PointMaze", **build_pointmaze_env_kwargs(env_config))


def run_analysis(args: argparse.Namespace) -> dict:
    env = _make_env()
    try:
        model = PathIntegrationRecurrentPPO.load(args.model_path, env=env, device=args.device)
        return compute_gridscore_analysis(
            model,
            args.dataset_root,
            batch_size_sequences=args.batch_size_sequences,
            max_seq_len=args.max_seq_len,
            n_bins=args.n_bins,
            bounds=None if args.bounds is None else tuple(float(v) for v in args.bounds),
            max_steps=args.max_steps,
            max_units=args.max_units,
            top_k=args.top_k,
            gridscore_positive_activations=args.gridscore_positive_activations,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size_sequences <= 0:
        parser.error("--batch-size-sequences must be > 0")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be > 0")
    if args.n_bins <= 0:
        parser.error("--n-bins must be > 0")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be > 0 when provided")
    if args.max_units is not None and args.max_units <= 0:
        parser.error("--max-units must be > 0 when provided")
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")

    output_dir = resolve_output_dir(args.model_path, args.dataset_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = run_analysis(args)
    write_analysis_config(args, output_dir, analysis["bounds"])
    save_gridscore_analysis(
        analysis,
        output_dir,
        data_filename="analysis_data.npz",
        summary_filename=None,
        top_k=args.top_k,
    )

    grid_scores = analysis["grid_scores"]
    valid = np.where(~np.isnan(grid_scores))[0]
    if valid.size:
        best = valid[np.argsort(grid_scores[valid])[::-1][0]]
        print(f"Best grid score: {grid_scores[best]:.6f} (unit {best})")
    else:
        print("No valid grid scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
