import os
import sys
from tqdm import tqdm
import time
from datetime import datetime
import mediapy as media
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

os.environ["MUJOCO_GL"] = "egl"
# paradir = os.path.dirname(curdir)
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)
import envs
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from utils.sb3_utils import random_action


if __name__ == "__main__":
    num_envs = 4
    # env = gym.make("DMCAnt-v0", task_name="Floor")

    # env = gym.make("DMCHumanoid-v0", task_name="Floor")

    # env = gym.make("DMCHumanoid-v0", task_name="Gaps")

    # env = gym.make("DMCHumanoid-v0", task_name="Maze")

    # env.reset()

    env_id = "DMCAnt-v0"
    task_name = "Maze"
    env = make_vec_env(
        env_id,
        n_envs=num_envs,
        env_kwargs={"task_name": task_name},
        vec_env_cls=SubprocVecEnv,
    )
    env.reset()
    print(env.observation_space)
    print(env.action_space)

    obs, reward, done, info = env.step(random_action(env.action_space.shape[0]))

    print(f"{obs.keys() = }")
    for k in obs.keys():
        print(f"{k} = {obs[k].shape}")
    print(reward)
    print(done)
    print(info)

    # speed test
    maze_map = []
    repeat_times = 100
    for i in tqdm(range(repeat_times)):
        obs, reward, done, info = env.step(random_action(env.action_space.shape[0]))
        if i % 20 == 0:
            env = make_vec_env(
                env_id,
                n_envs=num_envs,
                env_kwargs={"task_name": task_name},
                vec_env_cls=SubprocVecEnv,
            )
            # remake env will reset start postion and target of maze
            env.reset()
            maze_map += env.env_method(
                "custom_render", camera_id=0, width=128, height=128
            )

    def plot_map_list(num_envs, maze_map):
        # plot maze_map list in matplot, maze_map = [image, image, image]
        nrows = len(maze_map) // 4
        ncols = num_envs
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 10, nrows * 10),
        )

        # CRITICAL FIX: Flatten the 2D array of axes into a 1D list
        # This allows you to iterate through every subplot one by one
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            # Safety check to ensure we don't go out of bounds if grid > images
            if i < len(maze_map):
                # maze_map[i] shape is (128, 128, 3), which imshow handles automatically
                ax.imshow(
                    maze_map[i].astype("uint8")
                )  # Ensure type is uint8 or float for RGB
                ax.set_title(f"Env {i % num_envs} Map {i // num_envs}")
                ax.axis("off")
            else:
                # Turn off unused subplots
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"images/maze_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close(fig)

    plot_map_list(num_envs, maze_map)
