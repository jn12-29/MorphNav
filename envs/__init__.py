import os
import gymnasium as gym

gym.register(
    id="DMCAnt-v0",
    entry_point="custom_envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Ant"},
)

gym.register(
    id="DMCHumanoid-v0",
    entry_point="custom_envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Humanoid"},
)

gym.register(
    id="PointMaze",
    entry_point="custom_envs.point_maze:PointMazeEnv",
    max_episode_steps=300,
    # kwargs={"render_mode": "rgb_array"},
)
