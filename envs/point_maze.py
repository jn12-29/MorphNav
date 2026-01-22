"""A point mass maze environment with Gymnasium API.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Dict, List, Optional, Union

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from .mazes.maps import *
from .maze import MazeEnv
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class PointEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, xml_file: Optional[str] = None, **kwargs):

        self.cur_pos_aware = kwargs.pop("cur_pos_aware", True)
        self.sensor_aware = kwargs.pop("sensor_aware", True)
        if xml_file is None:
            xml_file = path.join(
                path.dirname(path.realpath(__file__)), "assets", "point.xml"
            )

        super().__init__(
            model_path=xml_file,
            frame_skip=1,
            observation_space=None,
            **kwargs,
        )
        if self.cur_pos_aware:
            obs_size = self.data.qpos.size
        else:
            obs_size = 0
        obs_size += self.data.qvel.size
        if self.data.sensordata.size > 0 and self.sensor_aware:
            obs_size += self.data.sensordata.size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        obs, _ = self._get_obs()

        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._clip_velocity()
        self.do_simulation(action, self.frame_skip)
        obs, info = self._get_obs()
        # This environment class has no intrinsic task, thus episodes don't end and there is no reward
        reward = 0
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs_list = [self.data.qvel]
        if self.cur_pos_aware:
            obs_list.append(self.data.qpos)
        if self.data.sensordata.size > 0 and self.sensor_aware:
            obs_list.append(self.data.sensordata)
        return np.concatenate(obs_list).ravel(), {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "sensordata": self.data.sensordata.copy(),
        }

    def _clip_velocity(self):
        """The velocity needs to be limited because the ball is
        force actuated and the velocity can grow unbounded."""
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)


class PointMazeEnv(MazeEnv, EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        maze_map_name: Optional[str] = None,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        cur_pos_aware: bool = True,
        target_aware: bool = True,
        sensor_aware: bool = True,
        time_penalty: float = 0.0,
        **kwargs,
    ):
        self.cur_pos_aware = cur_pos_aware
        self.target_aware = target_aware
        point_xml_file_path = kwargs.pop(
            "xml_file_path",
            path.join(path.dirname(path.realpath(__file__)), "assets", "point.xml"),
        )
        print("Loading point maze from XML file:", point_xml_file_path)
        if maze_map_name is not None:
            maze_map = eval(maze_map_name)
            print("Using Maze Map:", maze_map_name)
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            time_penalty=time_penalty,
            **kwargs,
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            cur_pos_aware=cur_pos_aware,
            sensor_aware=sensor_aware,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=(
                    spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64")
                    if cur_pos_aware
                    else spaces.Box(0, 0, shape=(1,), dtype="float64")
                ),
                desired_goal=(
                    spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64")
                    if target_aware
                    else spaces.Box(0, 0, shape=(1,), dtype="float64")
                ),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed=seed, **kwargs)
        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs, self.reset_pos)
        info["success"] = bool(np.linalg.norm(self.reset_pos - self.goal) <= 0.45)

        return obs_dict, info

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        cur_pos = info["qpos"][:2]
        obs_dict = self._get_obs(obs, cur_pos)
        reward = self.compute_reward(cur_pos, self.goal, info)
        terminated = self.compute_terminated(cur_pos, self.goal, info)
        truncated = self.compute_truncated(cur_pos, self.goal, info)
        info["success"] = bool(np.linalg.norm(cur_pos - self.goal) <= 0.45)

        # Update the goal position if necessary
        self.update_goal(cur_pos)

        return obs_dict, reward, terminated, truncated, info

    def update_target_site_pos(self):
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def _get_obs(self, point_obs, cur_pos) -> Dict[str, np.ndarray]:
        return {
            "observation": point_obs.copy(),
            "achieved_goal": (
                cur_pos.copy()
                if self.cur_pos_aware
                else np.array([0], dtype=np.float64)
            ),
            "desired_goal": (
                self.goal.copy()
                if self.target_aware
                else np.array([0], dtype=np.float64)
            ),
        }

    def render(self):
        return self.point_env.render()

    def close(self):
        super().close()
        self.point_env.close()

    @property
    def model(self):
        return self.point_env.model

    @property
    def data(self):
        return self.point_env.data
