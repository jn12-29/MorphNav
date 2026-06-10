from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch as th

from components.pi_eval_callback import OnlinePIEvalVisualizationCallback
from components.pi_eval_visualization import export_online_pi_eval_visualization


class _FakeTargetEncoder:
    def __init__(self) -> None:
        self.centers = th.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=th.float32,
        )


class _FakePolicy:
    def __init__(self) -> None:
        self.training = True
        self.path_integration_target_encoder = _FakeTargetEncoder()

    def set_training_mode(self, mode: bool) -> None:
        self.training = bool(mode)


class _FakeModel:
    def __init__(self) -> None:
        self.policy = _FakePolicy()

    def predict_with_pi(self, obs, state=None, episode_start=None, deterministic: bool = True):
        target_xy = np.asarray(obs["achieved_goal"], dtype=np.float32)
        centers = self.policy.path_integration_target_encoder.centers.numpy()
        logits = -np.sum((target_xy[:, None, :] - centers[None, :, :]) ** 2, axis=-1) * 12.0
        bottleneck = np.stack(
            [
                target_xy[:, 0] - 0.5,
                target_xy[:, 1] - 0.5,
                target_xy[:, 0] + target_xy[:, 1],
            ],
            axis=-1,
        ).astype(np.float32)
        actions = np.zeros((target_xy.shape[0], 2), dtype=np.float32)
        return actions, state, {"pc_logits": logits.astype(np.float32), "bottleneck": bottleneck}


class _FakeVecEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return self._obs()

    def step(self, actions):
        self.step_idx += 1
        done = self.step_idx >= 4
        obs = self._obs()
        if done:
            self.step_idx = 0
        return obs, np.zeros((1,), dtype=np.float32), np.asarray([done]), [{}]

    def _obs(self):
        xy = np.asarray([[float(self.step_idx % 2), float(self.step_idx // 2)]], dtype=np.float32)
        return {
            "observation": np.zeros((1, 2), dtype=np.float32),
            "start_pos": np.zeros((1, 2), dtype=np.float32),
            "achieved_goal": xy,
            "desired_goal": np.ones((1, 2), dtype=np.float32),
        }


def test_export_online_pi_eval_visualization_writes_eval_artifacts(tmp_path):
    summary = export_online_pi_eval_visualization(
        _FakeModel(),
        _FakeVecEnv(),
        tmp_path,
        n_eval_episodes=1,
        n_bins=4,
        top_k=2,
        max_total_steps=8,
    )

    assert summary["num_episodes"] == 1
    assert summary["num_steps"] == 4
    assert summary["localization"]["mse"] >= 0.0
    assert (tmp_path / "pi_eval_data.npz").is_file()
    assert (tmp_path / "pi_eval_summary.json").is_file()
    assert (tmp_path / "trajectory_preview.png").is_file()
    assert (tmp_path / "top_grid_cells.png").is_file()
    assert (tmp_path / "spatial_ratemaps_grid.png").is_file()

    payload = json.loads((tmp_path / "pi_eval_summary.json").read_text(encoding="utf-8"))
    assert payload["target_key"] == "achieved_goal"
    assert payload["gridscore"]["unit_count"] == 3
    assert payload["gridscore_positive_activations"] is False


def test_export_online_pi_eval_positive_activations_preserves_raw_bottleneck(tmp_path):
    summary = export_online_pi_eval_visualization(
        _FakeModel(),
        _FakeVecEnv(),
        tmp_path,
        n_eval_episodes=1,
        n_bins=4,
        top_k=2,
        max_total_steps=8,
        gridscore_positive_activations=True,
    )

    assert summary["gridscore_positive_activations"] is True
    assert summary["gridscore"]["gridscore_positive_activations"] is True
    with np.load(tmp_path / "pi_eval_data.npz") as data:
        assert np.min(data["bottleneck"]) < 0.0
        assert np.nanmin(data["ratemaps"]) >= 0.0


def test_online_pi_eval_callback_defaults_to_model_pi_target_key(tmp_path):
    callback = OnlinePIEvalVisualizationCallback(tmp_path)
    callback.model = SimpleNamespace(pi_target_key="custom_goal")

    assert callback._resolved_target_key() == "custom_goal"

    explicit = OnlinePIEvalVisualizationCallback(tmp_path, target_key="explicit_goal")
    explicit.model = SimpleNamespace(pi_target_key="custom_goal")

    assert explicit._resolved_target_key() == "explicit_goal"
