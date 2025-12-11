import os
import gymnasium as gym

gym.register(
    id="DMCAnt-v0",
    entry_point="custom_envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Ant"},
)
print(f"DMCAnt-v0 registered")

gym.register(
    id="DMCHumanoid-v0",
    entry_point="custom_envs.dmcenv:DMCEnv",
    max_episode_steps=1000,
    kwargs={"walker_name": "Humanoid"},
)
