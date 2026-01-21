import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    num_envs = 16
    env_id = "DMCAnt-v0"
    task_name = "Maze"
    env = make_vec_env(
        env_id,
        n_envs=num_envs,
        env_kwargs={"task_name": task_name},
        vec_env_cls=SubprocVecEnv,
    )
    model = PPO("MultiInputPolicy", env, n_epochs=100, verbose=2, device="cuda")
    model.learn(100_000)

    # save
    model.save(f"ppo_{env_id}")
