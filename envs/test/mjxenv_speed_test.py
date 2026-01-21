import os

os.environ["MUJOCO_GL"] = "egl"
from typing import Dict, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Discrete


if __name__ == "__main__":
    env = gym.make(
        "Ant-v5",
        render_mode="rgb_array",
        camera_name="egocentric",
        width=64,
        height=64,
        xml_file="/mnt/disk2/xh2/ai4neuron/derl/custom_envs/asset/ant_mjx.xml",
    )
    obs, info = env.reset()
    print(f"{obs = }")
    try:
        print(f"{obs.shape = }")
    except AttributeError:
        pass
    print(f"{env.action_space.shape = }")

    action_size = env.action_space.shape[0]

    def random_action():
        return np.random.uniform(-1, 1, (action_size))

    obs, reward, terminated, truncated, info = env.step(random_action())
    print(f"{reward = }")
    print(f"{terminated = }")
    print(f"{truncated = }")
    print(f"{info = }")
    repeat_times = 10_000

    from tqdm import tqdm
    import time

    st = time.time()
    for i in tqdm(range(repeat_times)):
        o, r, d, t, info = env.step(random_action())
        rgb_image = env.render()
    et = time.time()

    print(f"fps: {repeat_times / (et - st)}")
