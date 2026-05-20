import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "components" / "dataset_gen" / "pointmaze_policy.py"
SPEC = importlib.util.spec_from_file_location("pointmaze_policy", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load pointmaze_policy module at {MODULE_PATH}")
pointmaze_policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pointmaze_policy
SPEC.loader.exec_module(pointmaze_policy)

GridCellRandomWalkForceDriver = pointmaze_policy.GridCellRandomWalkForceDriver
PointMazePolicyConfig = pointmaze_policy.PointMazePolicyConfig


def _act(driver, xy, qvel=None, touch=None):
    if qvel is None:
        qvel = np.zeros(2, dtype=np.float32)
    return driver.act(agent_xy=xy, agent_qvel=qvel, touch=touch)


def test_grid_cell_force_driver_is_deterministic_for_seed():
    config = PointMazePolicyConfig(
        speed_mean=0.2,
        speed_std=0.0,
        angular_velocity_std=0.0,
        stuck_patience=100,
    )
    driver_a = GridCellRandomWalkForceDriver(config=config, seed=0)
    driver_b = GridCellRandomWalkForceDriver(config=config, seed=0)

    xy_a = np.array([0.0, 0.0], dtype=np.float32)
    xy_b = np.array([0.0, 0.0], dtype=np.float32)
    qvel_a = np.zeros(2, dtype=np.float32)
    qvel_b = np.zeros(2, dtype=np.float32)
    for _ in range(32):
        action_a = _act(driver_a, xy_a, qvel_a)
        action_b = _act(driver_b, xy_b, qvel_b)
        np.testing.assert_allclose(action_a, action_b)
        qvel_a = 0.5 * qvel_a + 0.1 * action_a
        qvel_b = 0.5 * qvel_b + 0.1 * action_b
        xy_a = xy_a + qvel_a * 0.01
        xy_b = xy_b + qvel_b * 0.01


def test_grid_cell_force_driver_explores_both_global_action_axes():
    config = PointMazePolicyConfig(
        speed_mean=0.3,
        speed_std=0.0,
        angular_velocity_std=1.6336,
        velocity_tracking_gain=4.0,
        action_smoothing=0.0,
        max_action_delta=1.0,
        stuck_patience=100,
    )
    driver = GridCellRandomWalkForceDriver(config=config, seed=3)

    xy = np.array([0.0, 0.0], dtype=np.float32)
    actions = []
    for _ in range(5000):
        action = _act(driver, xy)
        actions.append(action)
        xy = xy + action * 0.01

    action_arr = np.asarray(actions)
    assert action_arr[:, 0].min() < -0.2
    assert action_arr[:, 0].max() > 0.2
    assert action_arr[:, 1].min() < -0.2
    assert action_arr[:, 1].max() > 0.2


def test_grid_cell_force_driver_tracks_desired_velocity_with_force_feedback():
    config = PointMazePolicyConfig(
        speed_mean=0.2,
        speed_std=0.0,
        angular_velocity_std=0.0,
        velocity_tracking_gain=2.0,
        action_smoothing=0.0,
        max_action_delta=1.0,
        stuck_patience=100,
    )
    driver = GridCellRandomWalkForceDriver(config=config, seed=4)
    xy = np.array([0.0, 0.0], dtype=np.float32)

    accelerating_action = _act(driver, xy, qvel=np.zeros(2, dtype=np.float32))
    desired_velocity = driver._desired_velocity.copy()
    matched_action = _act(driver, xy + desired_velocity * 0.01, qvel=desired_velocity)

    assert float(np.dot(accelerating_action, desired_velocity)) > 0.0
    assert np.linalg.norm(matched_action) < np.linalg.norm(accelerating_action)


def test_grid_cell_force_driver_slides_from_touch_sensors_with_away_bias():
    config = PointMazePolicyConfig(
        speed_mean=0.2,
        speed_std=0.0,
        angular_velocity_std=0.0,
        velocity_tracking_gain=4.0,
        action_smoothing=0.0,
        max_action_delta=1.0,
        touch_tangent_weight=0.65,
        touch_away_weight=0.35,
        touch_jitter_angle=0.0,
    )
    driver = GridCellRandomWalkForceDriver(config=config, seed=0)
    driver._heading = 0.0

    right_wall_action = _act(
        driver,
        np.array([2.35, 0.0], dtype=np.float32),
        qvel=np.array([0.0, -0.1], dtype=np.float32),
        touch=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert right_wall_action[0] < 0.0
    assert right_wall_action[1] < 0.0

    driver._heading = 0.5 * np.pi
    top_wall_action = _act(
        driver,
        np.array([0.0, 2.35], dtype=np.float32),
        qvel=np.array([0.1, 0.0], dtype=np.float32),
        touch=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert top_wall_action[0] > 0.0
    assert top_wall_action[1] < 0.0


def test_grid_cell_force_driver_keeps_touch_jitter_deterministic_for_seed():
    config = PointMazePolicyConfig(
        speed_mean=0.2,
        speed_std=0.0,
        angular_velocity_std=0.0,
        velocity_tracking_gain=4.0,
        action_smoothing=0.0,
        max_action_delta=1.0,
    )
    driver_a = GridCellRandomWalkForceDriver(config=config, seed=12)
    driver_b = GridCellRandomWalkForceDriver(config=config, seed=12)

    action_a = _act(
        driver_a,
        np.array([2.35, 0.0], dtype=np.float32),
        qvel=np.array([0.0, -0.1], dtype=np.float32),
        touch=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    action_b = _act(
        driver_b,
        np.array([2.35, 0.0], dtype=np.float32),
        qvel=np.array([0.0, -0.1], dtype=np.float32),
        touch=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(action_a, action_b)


def test_default_stuck_threshold_is_below_normal_pointmaze_step_displacement():
    assert PointMazePolicyConfig().stuck_threshold < 0.001


def test_grid_cell_force_driver_resamples_when_stuck_for_patience_window():
    config = PointMazePolicyConfig(
        speed_mean=0.2,
        speed_std=0.0,
        angular_velocity_std=0.0,
        stuck_threshold=0.01,
        stuck_patience=2,
    )
    driver = GridCellRandomWalkForceDriver(config=config, seed=0)

    _act(driver, np.array([1.0, 0.0], dtype=np.float32))
    _act(driver, np.array([1.0, 0.0], dtype=np.float32))
    assert driver._stuck_counter == 1
    _act(driver, np.array([1.0, 0.0], dtype=np.float32))
    assert driver._stuck_counter == 0
