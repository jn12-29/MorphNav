import os

os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym

# from dm_control.locomotion.walkers import Rat
from custom_envs.dmc_walker import Rat
import dm_control.locomotion.arenas as arenas
import custom_envs.dmc_tasks as tasks
import numpy as np
from dm_control import composer
import shimmy
from gymnasium import Space, spaces
import sys
import cv2
import imageio


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    elif isinstance(observation_space, spaces.Box):
        return int(np.prod(observation_space.shape))
    elif isinstance(observation_space, spaces.Dict):
        return sum(get_flattened_obs_dim(subspace) for subspace in observation_space.spaces.values())
    else:
        raise NotImplementedError(f"Unsupported observation space: {observation_space}")


def count_obs_dim(gym_env: gym.Env):
    total_obs_dim = 0
    for key, subspace in gym_env.observation_space.items():
        total_obs_dim += get_flattened_obs_dim(subspace)
    return total_obs_dim


class DMCRatEnv(shimmy.DmControlCompatibilityV0):
    def __init__(self, task_name="gaps", seed=0, render_mode=None, **render_kwargs):
        walker = Rat(foot_mods=True)
        if task_name == "gaps":
            arena = arenas.GapsCorridor(
                platform_length=1.0,
                gap_length=0.1,
                corridor_width=2,
                corridor_length=10,
                visible_side_planes=True,
                aesthetic="outdoor_natural",
            )
            task = tasks.RunThroughCorridor(
                walker=walker,
                arena=arena,
                walker_spawn_position=(1, 0, 0),
                physics_timestep=0.001,
                control_timestep=0.005,
                forward_weight=2.0,
                healthy_weight=2.0,
                contact_termination=False,
                healthy_termination=True,
                healthy_z_range=(0.04, 0.1),
            )
        elif task_name == "escape":
            arena = arenas.Bowl(
                aesthetic="outdoor_natural",
            )
            task = tasks.Escape(
                walker=walker,
                arena=arena,
                walker_spawn_position=(0, 0, 0),
                physics_timestep=0.001,
                control_timestep=0.005,
            )
        else:
            raise NotImplementedError(f"Unsupported task: {task_name}")

        env = composer.Environment(
            task=task,
            time_limit=2,
            random_state=np.random.RandomState(seed),
            strip_singleton_obs_buffer_dim=True,
            recompile_mjcf_every_episode=False,
            fixed_initial_state=True,
        )

        super().__init__(env, render_mode, **render_kwargs)

        self.render_kwargs["camera_id"] = 1
        self.metadata["render_fps"] = 30

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(count_obs_dim(self),), dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # concatenate multiInput dict to one input array
        new_obs = np.concatenate([obs[key].flatten() for key in obs.keys()], axis=0)
        return new_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # concatenate multiInput dict to one input array
        new_obs = np.concatenate([obs[key].flatten() for key in obs.keys()], axis=0)
        return new_obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = DMCRatEnv(task_name="gaps")
    obs, info = env.reset()
    print(obs.shape)
    print(env.action_space.shape)
    action = np.random.uniform(-1, 1, (env.action_space.shape[0]))
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape)
    print(reward)
    print(terminated)
    print(truncated)
    print(info)


    # cp_print_stats = "tottime"
    # # cp_print_stats = 'cumtime'

    # env = DMCRatEnv(task_name="gaps", render_mode="rgb_array")
    # obs, info = env.reset()

    # # render and save image
    # image = env.render()
    # print(image)
    # cv2.imwrite("image.png", image)

    # print(f"obs shape: {obs.shape}")

    # action_size = env.action_space.shape[0]
    # print(f"action size: {action_size}")

    # def random_action():
    #     return np.random.uniform(-1, 1, (action_size))

    # repeat_times = 100

    # # render and save gif
    # videos = []
    # from tqdm import tqdm
    # import time

    # st = time.time()
    # import cProfile

    # start_time = time.time()
    # with cProfile.Profile() as cp:
    #     for i in tqdm(range(repeat_times)):
    #         o, r, d, t, info = env.step(random_action())
    #         videos.append(env.render())
    #     et = time.time()
    #     imageio.mimsave("output.gif", videos, fps=30)
    #     with open("dmc_phy_" + cp_print_stats + ".log", "w") as f:
    #         sys.stdout = f
    #         print(f"fps: {repeat_times / (et - st)}")
    #         # cp.print_stats('cumtime')
    #         cp.print_stats(cp_print_stats)
