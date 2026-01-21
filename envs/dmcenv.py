import os

os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym

from dm_control.locomotion.walkers import (
    Ant,
    CMUHumanoidPositionControlledV2020,
    JumpingBallWithHead,
    RollingBallWithHead,
)
import dm_control.locomotion.arenas as arenas
import dm_control.locomotion.tasks as tasks
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
        return sum(
            get_flattened_obs_dim(subspace)
            for subspace in observation_space.spaces.values()
        )
    else:
        raise NotImplementedError(f"Unsupported observation space: {observation_space}")


def count_obs_dim(gym_env: gym.Env, ignore_keys=["walker/egocentric_camera"]):
    total_obs_dim = 0
    for key, subspace in gym_env.observation_space.items():
        if key in ignore_keys:
            continue
        total_obs_dim += get_flattened_obs_dim(subspace)
    return total_obs_dim


class DMCEnv(shimmy.DmControlCompatibilityV0):
    def __init__(
        self,
        walker_name="Ant",
        task_name="Floor",
        walker_kwargs={},
        arena_kwargs={},
        task_kwargs={},
        env_kwargs={},
        seed=0,
        render_mode=None,
        **render_kwargs,
    ):
        # print(f"{walker_name =}, {task_name = }")
        if walker_name == "Ant":
            walker = Ant(**walker_kwargs)
        elif walker_name == "Humanoid":
            walker = CMUHumanoidPositionControlledV2020(**walker_kwargs)
        else:
            raise NotImplementedError(f"Unsupported walker: {walker_name}")

        if task_name == "Floor":
            arena = arenas.Floor(
                aesthetic="outdoor_natural",
                **arena_kwargs,
            )
            task = tasks.RunThroughCorridor(
                walker=walker,
                arena=arena,
                physics_timestep=0.001,
                control_timestep=0.005,
                contact_termination=False,
                **task_kwargs,
            )
        elif task_name == "Gaps":
            arena = arenas.GapsCorridor(
                # platform_length=2.0,
                # gap_length=0.1,
                # corridor_width=2,
                # corridor_length=10,
                visible_side_planes=True,
                aesthetic="outdoor_natural",
                **arena_kwargs,
            )
            task = tasks.RunThroughCorridor(
                walker=walker,
                arena=arena,
                walker_spawn_position=(1, 0, 0),  # will falldown without this
                physics_timestep=0.001,
                control_timestep=0.005,
                contact_termination=False,
                **task_kwargs,
            )
        elif task_name == "Escape":
            arena = arenas.Bowl(
                aesthetic="outdoor_natural",
                **arena_kwargs,
            )
            task = tasks.Escape(
                walker=walker,
                arena=arena,
                physics_timestep=0.001,
                control_timestep=0.005,
                **task_kwargs,
            )
        elif task_name == "Maze":
            from envs.mazes.build_labmaze import maze

            arena = arenas.MazeWithTargets(
                maze,
                xy_scale=2.0,
                z_height=2.0,
                skybox_texture=None,
                wall_textures=None,
                floor_textures=None,
                aesthetic="outdoor_natural",
                name="maze",
                **arena_kwargs,
            )
            task = tasks.RepeatSingleGoalMaze(
                walker=walker,
                maze_arena=arena,
                **task_kwargs,
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

        # self.render_kwargs["camera_id"] = 1
        # self.metadata["render_fps"] = 30

        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(count_obs_dim(self),), dtype=np.float64
        # )

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(count_obs_dim(self),),
                    dtype=np.float64,
                ),
            }
        )

        self._debug_mode = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self._obs_process(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._obs_process(obs)
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def debug(self, opt):
        self._debug_mode = opt

    def _obs_process(self, obs: dict):
        # egocentric_camera
        egocentric = obs.pop("walker/egocentric_camera")
        # concatenate multiInput dict to one input array
        proprioception = np.concatenate(
            [obs[key].flatten() for key in obs.keys()], axis=0
        )
        return {"image": egocentric, "proprio": proprioception}

    def custom_render(self, camera_id=0, width=128, height=128):
        return self._env.physics.render(camera_id=camera_id, width=width, height=height)


if __name__ == "__main__":
    wn = "Ant"
    tn = "Gaps"
    tn = "Floor"
    env = DMCEnv(walker_name=wn, task_name=tn)
    env.debug(True)
    obs, info = env.reset()
    print(f"{obs = }")
    print(f"{env.action_space.shape = }")

    action_size = env.action_space.shape[0]
    print(f"action size: {action_size}")

    def random_action():
        return np.random.uniform(-1, 1, (action_size))

    obs, reward, terminated, truncated, info = env.step(random_action())
    print(f"{reward = }")
    print(f"{terminated = }")
    print(f"{truncated = }")
    print(f"{info = }")

    from tqdm import tqdm
    import time

    videos_egocentric = []
    videos_side = []
    repeat_times = 300
    camera_id = 2
    st = time.time()
    for i in tqdm(range(repeat_times)):
        obs, reward, terminated, truncated, info = env.step(random_action())
        videos_egocentric.append(obs["image"])
        videos_side.append(
            env._env.physics.render(camera_id=camera_id, width=128, height=128)
        )
        if i % 100 == 0:
            obs, info = env.reset()

    et = time.time()

    print(f"fps: {repeat_times / (et - st)}")
    imageio.mimsave(f"{wn}_{tn}_egocentric_camera.gif", videos_egocentric, fps=30)
    imageio.mimsave(f"{wn}_{tn}_camera_id_{camera_id}.gif", videos_side, fps=30)

    # cp_print_stats = "tottime"
    # # cp_print_stats = 'cumtime'

    # env = DMCRatEnv(task_name="gaps", render_mode="rgb_array")
    # obs, info = env.reset()

    # # render and save image
    # image = env.render()
    # print(image)
    # cv2.imwrite("image.png", image)

    # print(f"obs shape: {obs.shape}")

    # repeat_times = 100

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
