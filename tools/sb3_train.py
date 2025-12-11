import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import custom_envs
from stable_baselines3 import PPO

from utils.sb3_utils import make_env, make_subproc_vec_env


if __name__ == "__main__":
    num_envs = 16
    env_id = "DMCAnt-v0"
    task_name = "Maze"
    env = make_subproc_vec_env(env_id, task_name, num_envs)
    model = PPO("MultiInputPolicy", env, n_epochs=100, verbose=2, device="cuda")
    model.learn(100_000)

    # save
    model.save(f"ppo_{env_id}")
