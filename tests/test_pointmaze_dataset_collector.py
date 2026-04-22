import numpy as np
import pytest

from components.dataset_gen.pointmaze_collector import collect_episode
from components.dataset_gen.pointmaze_episode import summarize_episode


class FakeEnv:
    def __init__(self) -> None:
        self.step_idx = 0
        self.goal = np.array([2.0, 2.0], dtype=np.float32)
        self.reset_seed = None

    def reset(self, seed=None):
        self.step_idx = 0
        self.reset_seed = seed
        obs = {"observation": np.array([0.0], dtype=np.float32), "desired_goal": self.goal.copy()}
        info = {
            "qpos": np.array([0.0, 0.0], dtype=np.float32),
            "qvel": np.array([0.0, 0.0], dtype=np.float32),
            "success": False,
            "collision": False,
        }
        return obs, info

    def step(self, action):
        self.step_idx += 1
        qpos = np.array([float(self.step_idx), 0.0], dtype=np.float32)
        qvel = np.array([1.0, 0.0], dtype=np.float32)
        obs = {"observation": np.array([self.step_idx], dtype=np.float32), "desired_goal": self.goal.copy()}
        info = {
            "qpos": qpos,
            "qvel": qvel,
            "success": self.step_idx >= 3,
            "collision": False,
            "speed": 1.0,
        }
        return obs, 0.5, self.step_idx >= 3, False, info


class ConstantPolicy:
    def act(self, agent_xy, heading, collision):
        del agent_xy, heading, collision
        return np.array([0.5, 0.0], dtype=np.float32)


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
    assert episode["action"].shape == (3, 2)
    assert episode["qpos"].shape == (3, 2)
    assert episode["summary"]["episode_length"] == 3
    assert episode["summary"]["path_length"] == 2.0
    assert episode["summary"]["goal_reached"] is True


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


def test_collect_episode_requires_qpos_in_reset_info():
    class MissingResetQposEnv(FakeEnv):
        def reset(self, seed=None):
            obs, info = super().reset(seed=seed)
            del info["qpos"]
            return obs, info

    with pytest.raises(KeyError, match="Missing required field in reset info: 'qpos'"):
        collect_episode(
            env=MissingResetQposEnv(),
            policy=ConstantPolicy(),
            episode_id=0,
            episode_seed=1,
            env_metadata={},
            policy_metadata={},
        )


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


def test_collect_episode_requires_desired_goal_in_step_obs():
    class MissingGoalObsEnv(FakeEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            del obs["desired_goal"]
            return obs, reward, terminated, truncated, info

    with pytest.raises(KeyError, match="Missing required field in step observation: 'desired_goal'"):
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
