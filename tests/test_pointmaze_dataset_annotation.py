import importlib.util
from pathlib import Path
import sys

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "components" / "dataset_gen" / "pointmaze_annotation.py"
SPEC = importlib.util.spec_from_file_location("pointmaze_annotation", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load pointmaze_annotation module at {MODULE_PATH}")
pointmaze_annotation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pointmaze_annotation
SPEC.loader.exec_module(pointmaze_annotation)

annotate_episode = pointmaze_annotation.annotate_episode


def test_annotate_episode_derives_stable_spatial_fields():
    episode = {
        "qpos": np.array([[1.0, 2.0, 0.0], [1.5, 2.5, 0.1]], dtype=np.float64),
        "goal": np.array([[3.0, 5.0, 9.0], [3.0, 5.0, 9.0]], dtype=np.float64),
    }

    annotations = annotate_episode(episode)

    np.testing.assert_allclose(annotations["agent_xy"], np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32))
    np.testing.assert_allclose(annotations["goal_xy"], np.array([[3.0, 5.0], [3.0, 5.0]], dtype=np.float32))
    np.testing.assert_allclose(
        annotations["relative_goal"],
        np.array([[2.0, 3.0], [1.5, 2.5]], dtype=np.float32),
    )


def test_heading_uses_position_deltas_and_stabilizes_first_element():
    episode = {
        "qpos": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [1.0, 1.0, 0.2],
                [2.0, 1.0, 0.3],
            ],
            dtype=np.float64,
        ),
        "goal": np.zeros((4, 2), dtype=np.float64),
    }

    heading = annotate_episode(episode)["heading"]

    assert np.isclose(heading[0], np.pi / 4)
    assert np.isclose(heading[2], np.pi / 4)
    assert np.isclose(heading[3], 0.0)


def test_annotate_episode_broadcasts_fixed_goal_to_timestep_shape():
    episode = {
        "qpos": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float64),
        "goal": np.array([3.0, 5.0], dtype=np.float64),
    }

    annotations = annotate_episode(episode)

    np.testing.assert_allclose(
        annotations["goal_xy"],
        np.array([[3.0, 5.0], [3.0, 5.0], [3.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        annotations["relative_goal"],
        np.array([[3.0, 4.0], [2.0, 3.0], [1.0, 1.0]], dtype=np.float32),
    )
