import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import custom_envs
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(env_id, rank, seed=0):
    """
    用于创建并返回一个环境实例的函数，必须定义在全局作用域。
    """

    def _init():
        env = gym.make(env_id, seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    num_envs = 32  # 并行环境数量，建议为 CPU 核心数的 1.2~1.5 倍
    env_id = "DMCAnt-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    model = PPO("MultiInputPolicy", env, n_epochs=100, verbose=2, device="cuda")
    model.learn(100_000)

    # save
    model.save(f"ppo_{env_id}")
