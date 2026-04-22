# AGENTS.md

This file guides coding agents when working in this repository.

## Repo Summary

MorphNav is a maze-navigation reinforcement learning research repository.
Most project-specific code lives in custom environments under `envs/` and auxiliary policy code under `components/`.
Training and evaluation rely on the local customized `rl-baselines3-zoo/` tree.
Treat many shell scripts as experiment notes or starting points, not polished automation.

## Key Paths

- `envs/`: custom Gymnasium/MuJoCo environments and environment registration.
- `components/`: auxiliary recurrent-policy and feature-extractor code.
- `scripts/`: setup and example experiment commands.
- `tests/`: smoke tests, probes, and analysis scripts.
- `rl-baselines3-zoo/`: local training stack, configs, and recording utilities used by this repo.

## Typical Commands

Use these as starting points and inspect the related script before running variants.

```bash
conda create -n mz python=3.12 -y
conda run -n mz pip install -r requirements.txt
pip install -e ./rl-baselines3-zoo

python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml

tensorboard --logdir logs/

python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id <ID> -n 1000 --load-best

python tests/point_maze_env_test.py
python tests/ant_maze_env_test.py
```

## Agent Rules

- Read `scripts/` before inventing new training or extraction commands.
- Treat `scripts/*.sh` as examples; some contain hard-coded devices or paths.
- Avoid editing `rl-baselines3-zoo/` unless the task really requires training-stack changes.
- When changing env behavior, inspect `envs/__init__.py`, the env constructor, and any dependent scripts together.
- Keep edits minimal and run the smallest relevant validation.

## Gotchas

- `README.md` is incomplete and should not be treated as the full source of truth.
- Some shell scripts contain local absolute paths and fixed GPU ids.
- Some files under `tests/` are exploratory analysis scripts, not strict automated tests.
- The local `rl-baselines3-zoo/` may differ from upstream behavior and APIs.
