# CLAUDE.md

This file lists MorphNav-specific traps that Claude Code tends to miss.

## Rules

- `README.md` is the user entry point, but not the full source of truth. Inspect scripts and code before changing commands or behavior.
- Run project commands in the conda `mz` environment. If the tool shell does not inherit the user's active shell environment, use `conda run -n mz ...` explicitly.
- Pytest can appear to hang under the sandbox for this project. If a test run stalls without output, check for sandbox-related blockage and rerun the smallest relevant test selection outside the sandbox with approval instead of assuming the tests are broken.
- Do not create or maintain `AGENTS.md`. The old `docs/superpowers/` plugin documents are not source-of-truth docs.
- `rl-baselines3-zoo/` is the local training stack and a git submodule. Avoid editing it unless the task requires training-stack changes, and check submodule status when it changes.
- Treat `scripts/*.sh` as experiment notes. They may contain fixed GPU ids, local absolute paths, and stale command variants.
- When changing environment behavior, inspect `envs/__init__.py`, the environment constructor, related zoo configs, and scripts together.
- Phase 1 dataset and offline-PI contracts live near the implementation in `components/dataset_gen/`, `components/offline_pi_rehearsal.py`, and `components/pi_*.py`; inspect those modules before changing behavior.
- Keep edits surgical and run the smallest relevant validation.
