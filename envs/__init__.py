import os
import gymnasium as gym

gym.register(
    id="DMCAnt-v0",
    entry_point="envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Ant"},
)

gym.register(
    id="DMCHumanoid-v0",
    entry_point="envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Humanoid"},
)

gym.register(
    id="PointMaze",
    entry_point="envs.point_maze:PointMazeEnv",
    max_episode_steps=300,
)

gym.register(
    id="AntMaze",
    entry_point="envs.ant_maze:AntMazeEnv",
    max_episode_steps=500,
)
