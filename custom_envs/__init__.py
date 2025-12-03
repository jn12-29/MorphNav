import os
import gymnasium as gym

gym.register(
    'mujoco-Rat-v0',
    entry_point= 'custom_envs.gym_mujocoenv:MujocoRatEnv',
    max_episode_steps=2000
)

gym.register(
    id='dmc-Rat-gaps-v0',
    entry_point='custom_envs.gym_dmcenv:DMCRatEnv',
    kwargs={
        'task_name': 'gaps'
    }
)

gym.register(
    id='dmc-Rat-escape-v0',
    entry_point='custom_envs.gym_dmcenv:DMCRatEnv',
    kwargs={
        'task_name': 'escape'
    }
)
