# CLAUDE.md

This file lists MorphNav-specific traps that Claude Code tends to miss.

## Rules

- `README.md` is the user entry point, but not the full source of truth. Inspect scripts and code before changing commands or behavior.
- Run project commands in the conda `mz` environment. If the tool shell does not inherit the user's active shell environment, use `conda run -n mz ...` explicitly.
- Pytest can appear to hang under the sandbox for this project. If a test run stalls without output, check for sandbox-related blockage and rerun the test outside the sandbox instead of assuming the tests are broken.
- Do not create or maintain `AGENTS.md`. The old `docs/superpowers/` plugin documents are not source-of-truth docs.
- `rl-baselines3-zoo/` is the local training stack and a git submodule. Avoid editing it unless the task requires training-stack changes, and check submodule status when it changes.
- Treat `scripts/*.sh` as experiment notes. They may contain fixed GPU ids, local absolute paths, and stale command variants.
- New data and output paths should follow the project layout: reusable datasets under `data/datasets/`; experiment outputs, models, TensorBoard logs, rollout recordings, metrics, and analysis artifacts under `runs/`.
- When changing environment behavior, inspect `envs/__init__.py`, the environment constructor, related zoo configs, and scripts together.
- Phase 1 dataset and offline-PI contracts live near the implementation in `components/dataset_gen/`, `components/offline_pi_rehearsal.py`, and `components/pi_*.py`; inspect those modules before changing behavior.
- Keep Phase 1 fresh-model settings in `rl-baselines3-zoo/conf/maze_pi.yml`; standalone offline PI scripts should load from that config instead of duplicating model kwargs.
- Standalone offline PI runs write a full run directory under `runs/offline_pi/`: keep `train.log`, `config.json`, `metrics/metrics.jsonl`, compatibility metrics, checkpoints, and probe artifacts aligned when changing that workflow.
- Keep `scripts/pi.sh` aligned with the standalone offline PI CLI and README examples.
- Keep edits surgical and run the smallest relevant validation.
