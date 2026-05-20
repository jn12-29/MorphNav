import numpy as np
import pytest

from components.dataset_gen.pointmaze_annotation import annotate_episode
from components.dataset_gen.pointmaze_collector import collect_episode
from components.dataset_gen.pointmaze_config import PointMazeEnvConfig
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs, create_pointmaze_env
from components.dataset_gen.pointmaze_episode import summarize_episode


class FakeEnv:
    def __init__(self) -> None:
        self.step_idx = 0
        self.goal = np.array([2.0, 2.0], dtype=np.float32)
        self.reset_seed = None

    def _obs(self, step_idx: int) -> dict[str, np.ndarray]:
        xy = np.array([float(step_idx), 0.0], dtype=np.float32)
        return {
            "observation": np.array([float(step_idx)], dtype=np.float32),
            "start_pos": np.array([0.0, 0.0], dtype=np.float32),
            "achieved_goal": xy,
            "desired_goal": self.goal.copy(),
        }

    def reset(self, seed=None):
        self.step_idx = 0
        self.reset_seed = seed
        obs = self._obs(0)
        info = {
            "qpos": np.array([0.0, 0.0], dtype=np.float32),
            "qvel": np.array([0.0, 0.0], dtype=np.float32),
            "sensordata": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "success": False,
        }
        return obs, info

    def step(self, action):
        self.step_idx += 1
        qpos = np.array([float(self.step_idx), 0.0], dtype=np.float32)
        qvel = np.array([1.0, 0.0], dtype=np.float32)
        obs = self._obs(self.step_idx)
        info = {
            "qpos": qpos,
            "qvel": qvel,
            "sensordata": np.array([float(self.step_idx), 0.0, 0.0, 0.0], dtype=np.float32),
            "success": self.step_idx >= 3,
            "speed": 1.0,
        }
        return obs, 0.5, self.step_idx >= 3, False, info


class ConstantPolicy:
    def act(self, agent_xy, agent_qvel, touch):
        del agent_xy, agent_qvel, touch
        return np.array([0.5, 0.0], dtype=np.float32)


class RecordingPolicy:
    def __init__(self) -> None:
        self.calls = []

    def act(self, agent_xy, agent_qvel, touch):
        self.calls.append(
            {
                "agent_xy": np.asarray(agent_xy, dtype=np.float32).copy(),
                "agent_qvel": np.asarray(agent_qvel, dtype=np.float32).copy(),
                "touch": np.asarray(touch, dtype=np.float32).copy(),
            }
        )
        return np.array([0.5, 0.0], dtype=np.float32)


class NeverDoneEnv(FakeEnv):
    def step(self, action):
        obs, reward, _terminated, _truncated, info = super().step(action)
        return obs, reward, False, False, info


def test_build_pointmaze_env_kwargs_and_create_env(monkeypatch):
    config = PointMazeEnvConfig(
        maze_map_name="U_MAZE",
        continuing_task=False,
        reset_target=True,
        max_episode_steps=321,
        sensor_aware=True,
        achieved_goal_aware=True,
        start_pos_aware=True,
        target_aware=True,
        xml_file_path="/tmp/point_v1.xml",
        success_radius=0.4,
    )
    kwargs = build_pointmaze_env_kwargs(config)

    assert kwargs == {
        "maze_map_name": "U_MAZE",
        "continuing_task": False,
        "reset_target": True,
        "sensor_aware": True,
        "achieved_goal_aware": True,
        "start_pos_aware": True,
        "target_aware": True,
        "xml_file_path": "/tmp/point_v1.xml",
        "success_radius": 0.4,
    }
    assert "max_episode_steps" not in kwargs

    captured = {}

    class DummyPointMazeEnv:
        def __init__(self, **init_kwargs):
            captured.update(init_kwargs)

    monkeypatch.setattr(
        "components.dataset_gen.pointmaze_env_factory._resolve_pointmaze_env_cls",
        lambda: DummyPointMazeEnv,
    )
    create_pointmaze_env(config)
    assert captured == kwargs


def test_pointmaze_env_config_default_maze_map_name_is_uppercase_constant():
    config = PointMazeEnvConfig()
    assert config.maze_map_name == "OPEN"


def test_collect_episode_returns_normalized_episode():
    env = FakeEnv()
    episode = collect_episode(
        env=env,
        policy=ConstantPolicy(),
        episode_id=4,
        episode_seed=101,
        env_metadata={"maze_map_name": "U_MAZE"},
        policy_metadata={"policy_type": "constant"},
    )

    assert env.reset_seed == 101
    assert episode["episode_id"] == 4
    assert episode["seed"] == 101
    assert episode["obs"]["observation"].shape == (3, 1)
    np.testing.assert_array_equal(episode["obs"]["observation"].reshape(-1), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(
        episode["obs"]["achieved_goal"],
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    assert episode["action"].shape == (3, 2)
    assert episode["qpos"].shape == (3, 2)
    assert episode["summary"]["episode_length"] == 3
    assert episode["summary"]["path_length"] == 2.0
    assert episode["summary"]["goal_reached"] is True


def test_collect_episode_passes_current_xy_and_qvel_to_policy():
    policy = RecordingPolicy()

    collect_episode(
        env=FakeEnv(),
        policy=policy,
        episode_id=4,
        episode_seed=101,
        env_metadata={"maze_map_name": "U_MAZE"},
        policy_metadata={"policy_type": "recording"},
    )

    assert len(policy.calls) == 3
    np.testing.assert_array_equal(policy.calls[0]["agent_xy"], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[0]["agent_qvel"], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[0]["touch"], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[1]["agent_xy"], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[1]["agent_qvel"], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[1]["touch"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_collect_episode_can_be_annotated():
    episode = collect_episode(
        env=FakeEnv(),
        policy=ConstantPolicy(),
        episode_id=10,
        episode_seed=33,
        env_metadata={},
        policy_metadata={},
    )
    annotation = annotate_episode(episode)

    assert annotation["agent_xy"].shape == (3, 2)
    assert annotation["heading"].shape == (3,)
    assert annotation["goal_xy"].shape == (3, 2)
    assert annotation["relative_goal"].shape == (3, 2)


def test_collect_episode_enforces_max_episode_steps_cap_from_env_metadata():
    config = PointMazeEnvConfig(max_episode_steps=2)
    episode = collect_episode(
        env=NeverDoneEnv(),
        policy=ConstantPolicy(),
        episode_id=11,
        episode_seed=44,
        env_metadata={"max_episode_steps": config.max_episode_steps},
        policy_metadata={},
    )

    assert episode["summary"]["episode_length"] == 2
    assert episode["terminated"].tolist() == [False, False]
    assert episode["truncated"].tolist() == [False, True]


def test_summarize_episode_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="rewards has 3 steps but terminated has 2 steps"):
        summarize_episode(
            qpos=np.zeros((3, 2), dtype=np.float32),
            qvel=np.zeros((3, 2), dtype=np.float32),
            rewards=np.zeros((3,), dtype=np.float32),
            terminated=np.zeros((2,), dtype=bool),
            truncated=np.zeros((3,), dtype=bool),
            info_list=[{}, {}, {}],
        )


def test_collect_episode_uses_achieved_goal_when_reset_qpos_is_missing():
    class MissingResetQposEnv(FakeEnv):
        def reset(self, seed=None):
            obs, info = super().reset(seed=seed)
            del info["qpos"]
            return obs, info

    episode = collect_episode(
        env=MissingResetQposEnv(),
        policy=ConstantPolicy(),
        episode_id=0,
        episode_seed=1,
        env_metadata={},
        policy_metadata={},
    )
    assert episode["summary"]["episode_length"] == 3


def test_collect_episode_raises_when_observation_keys_change():
    class KeyDriftEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            obs["extra"] = np.array([1.0], dtype=np.float32)
            return obs, reward, terminated, truncated, info

    with pytest.raises(ValueError, match="Observation keys changed across steps"):
        collect_episode(
            env=KeyDriftEnv(),
            policy=ConstantPolicy(),
            episode_id=0,
            episode_seed=1,
            env_metadata={},
            policy_metadata={},
        )


def test_collect_episode_requires_policy_obs_keys_in_step_obs():
    class MissingGoalObsEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            del obs["desired_goal"]
            return obs, reward, terminated, truncated, info

    with pytest.raises(ValueError, match="Observation keys changed across steps"):
        collect_episode(
            env=MissingGoalObsEnv(),
            policy=ConstantPolicy(),
            episode_id=0,
            episode_seed=1,
            env_metadata={},
            policy_metadata={},
        )


def test_collect_episode_requires_qvel_in_step_info():
    class MissingQvelEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            del info["qvel"]
            return obs, reward, terminated, truncated, info

    with pytest.raises(KeyError, match="Missing required field in step info: 'qvel'"):
        collect_episode(
            env=MissingQvelEnv(),
            policy=ConstantPolicy(),
            episode_id=0,
            episode_seed=1,
            env_metadata={},
            policy_metadata={},
        )


def test_collect_episode_accepts_reordered_equivalent_observation_keys():
    class ReorderedObsEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            reordered = {
                "desired_goal": obs["desired_goal"],
                "achieved_goal": obs["achieved_goal"],
                "start_pos": obs["start_pos"],
                "observation": obs["observation"],
            }
            return reordered, reward, terminated, truncated, info

    episode = collect_episode(
        env=ReorderedObsEnv(),
        policy=ConstantPolicy(),
        episode_id=0,
        episode_seed=1,
        env_metadata={},
        policy_metadata={},
    )
    assert episode["obs"]["observation"].shape == (3, 1)
    assert episode["goal"].shape == (3, 2)
