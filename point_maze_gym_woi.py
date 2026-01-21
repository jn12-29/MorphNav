import gymnasium as gym
import os

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import envs
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    num_envs = 2
    env_id = "PointMaze"
    env = make_vec_env(env_id, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
    model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=2, device="cuda")
    model.learn(1_000_000)

    # save
    model.save(f"ppo_{env_id}")
