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

from mazes.maps import U_MAZE
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


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
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = False,
        reset_target: bool = False,
        **kwargs,
    ):
        point_xml_file_path = path.join(path.dirname(__file__), "assets", "point.xml")
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=0.4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=obs_shape, dtype="float64"
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
        obs = self._get_obs(obs)
        achieved_goal = obs[:2]
        info["success"] = bool(np.linalg.norm(achieved_goal - self.goal) <= 0.45)

        return obs, info

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        obs = self._get_obs(obs)
        achieved_goal = obs[:2]

        reward = self.compute_reward(achieved_goal, self.goal, info)
        terminated = self.compute_terminated(achieved_goal, self.goal, info)
        truncated = self.compute_truncated(achieved_goal, self.goal, info)
        info["success"] = bool(np.linalg.norm(achieved_goal - self.goal) <= 0.45)

        # Update the goal position if necessary
        self.update_goal(achieved_goal)

        return obs, reward, terminated, truncated, info

    def update_target_site_pos(self):
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def _get_obs(self, point_obs) -> np.ndarray:
        return point_obs.copy()

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
