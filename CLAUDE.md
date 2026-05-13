# CLAUDE.md

This file lists MorphNav-specific traps that Claude Code tends to miss.

## Rules

- `README.md` is the user entry point, but not the full source of truth. Inspect scripts and code before changing commands or behavior.
- Do not create or maintain `AGENTS.md`. The old `docs/superpowers/` plugin documents are not source-of-truth docs.
- `rl-baselines3-zoo/` is the local training stack and a git submodule. Avoid editing it unless the task requires training-stack changes, and check submodule status when it changes.
- Treat `scripts/*.sh` as experiment notes. They may contain fixed GPU ids, local absolute paths, and stale command variants.
- When changing environment behavior, inspect `envs/__init__.py`, the environment constructor, related zoo configs, and scripts together.
- The current PointMaze dataset contract is implemented in `scripts/generate_pointmaze_dataset.py` and `components/dataset_gen/`, not in old planning docs.
- `pi_ppo_lstm` depends on `components/pi_*.py`, `components/path_integration.py`, `rl-baselines3-zoo/conf/maze_pi.yml`, and zoo algorithm registration. `achieved_goal` is the PI target and should be dropped from policy features, not from rollout observations.
- Keep edits surgical and run the smallest relevant validation.
