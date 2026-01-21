import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


def random_action(action_size):
    return np.random.uniform(-1, 1, (action_size))
