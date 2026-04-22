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

WeakRandomPolicyDriver = pointmaze_policy.WeakRandomPolicyDriver
PointMazePolicyConfig = pointmaze_policy.PointMazePolicyConfig


def test_action_is_held_within_fixed_segment_length():
    config = PointMazePolicyConfig(
        segment_length_range=(3, 3),
        action_noise_scale=0.0,
        stuck_threshold=0.0,
        stuck_patience=100,
        subgoal_resample_prob=0.0,
        turn_bias=0.0,
        forward_bias=1.0,
    )
    driver = WeakRandomPolicyDriver(config=config, seed=0)

    action0 = driver.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    action1 = driver.act(agent_xy=np.array([10.0, 0.0]), heading=0.0, collision=False)
    action2 = driver.act(agent_xy=np.array([20.0, 0.0]), heading=0.0, collision=False)

    np.testing.assert_allclose(action1, action0)
    np.testing.assert_allclose(action2, action0)


def test_resamples_when_stuck_for_patience_window():
    config = PointMazePolicyConfig(
        segment_length_range=(100, 100),
        action_noise_scale=0.0,
        stuck_threshold=0.01,
        stuck_patience=2,
        subgoal_resample_prob=0.0,
        turn_bias=0.0,
        forward_bias=1.0,
    )
    driver = WeakRandomPolicyDriver(config=config, seed=0)

    scripted_actions = [
        np.array([0.8, 0.1], dtype=np.float32),
        np.array([-0.2, 0.5], dtype=np.float32),
        np.array([0.3, -0.7], dtype=np.float32),
    ]
    call_count = {"n": 0}

    def _scripted_sample() -> np.ndarray:
        action = scripted_actions[min(call_count["n"], len(scripted_actions) - 1)]
        call_count["n"] += 1
        return action.copy()

    driver._sample_action = _scripted_sample  # type: ignore[attr-defined]

    first = driver.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    second = driver.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    third = driver.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)

    np.testing.assert_allclose(second, first)
    np.testing.assert_allclose(third, np.array([-0.2, 0.5], dtype=np.float32))
    assert call_count["n"] == 2
