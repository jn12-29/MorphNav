from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pointmaze_config import PointMazeEnvConfig

if TYPE_CHECKING:
    from envs.point_maze import PointMazeEnv


def build_pointmaze_env_kwargs(config: PointMazeEnvConfig) -> dict[str, Any]:
    # max_episode_steps is intentionally excluded from env kwargs.
    # It is used as collector control metadata to cap rollout length.
    kwargs: dict[str, Any] = {
        "maze_map_name": config.maze_map_name,
        "continuing_task": config.continuing_task,
        "reset_target": config.reset_target,
        "sensor_aware": config.sensor_aware,
    }
    if config.xml_file_path:
        kwargs["xml_file_path"] = config.xml_file_path
    return kwargs


def _resolve_pointmaze_env_cls():
    from envs.point_maze import PointMazeEnv

    return PointMazeEnv


def create_pointmaze_env(config: PointMazeEnvConfig) -> PointMazeEnv:
    env_cls = _resolve_pointmaze_env_cls()
    return env_cls(**build_pointmaze_env_kwargs(config))
