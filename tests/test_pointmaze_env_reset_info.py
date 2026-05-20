from pathlib import Path

import numpy as np

from envs.point_maze import PointMazeEnv


def test_pointmaze_reset_info_includes_mujoco_state_and_sensors():
    xml_file_path = Path(__file__).resolve().parents[1] / "envs" / "assets" / "point_v1.xml"
    env = PointMazeEnv(
        maze_map_name="OPEN",
        xml_file_path=str(xml_file_path),
        sensor_aware=True,
        achieved_goal_aware=True,
        start_pos_aware=True,
        target_aware=True,
        continuing_task=True,
    )
    try:
        obs, info = env.reset(seed=0)
    finally:
        env.close()

    assert {"qpos", "qvel", "sensordata", "success"}.issubset(info.keys())
    assert np.asarray(info["qpos"]).shape[0] >= 2
    assert np.asarray(info["qvel"]).shape[0] >= 2
    assert np.asarray(info["sensordata"]).shape == (4,)
    assert obs["observation"].shape == (6,)
    np.testing.assert_array_equal(obs["observation"][-4:], info["sensordata"])
