import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv


def random_action(action_size):
    return np.random.uniform(-1, 1, (action_size))


def make_env(env_id, task_name, rank, seed=0):
    """
    用于创建并返回一个环境实例的函数，必须定义在全局作用域。
    """

    def _init():
        env = gym.make(env_id, task_name=task_name, seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_subproc_vec_env(env_id, task_name, num_envs, seed=0):
    """
    创建一个包含多个环境的向量环境。
    """
    return SubprocVecEnv(
        [make_env(env_id, task_name, i, seed) for i in range(num_envs)]
    )
