from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from components.aux_extractor import CustomCombinedExtractor
from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
)
from components.dataset_gen.pointmaze_zarr_writer import write_shard
from components.offline_pi_eval_artifacts import export_probe_artifacts
from components.offline_pi_gridscore import export_gridscore_artifacts
from components.offline_pi_rehearsal import (
    count_offline_pi_sequences,
    compute_offline_pi_loss,
    decode_offline_pi_coordinates,
    load_offline_pi_batches,
    make_offline_pi_optimizer,
    run_offline_pi_probe,
    run_offline_pi_rehearsal,
)
from components.pi_policy import PathIntegrationRecurrentActorCriticPolicy


def _make_policy() -> PathIntegrationRecurrentActorCriticPolicy:
    obs_space = spaces.Dict(
        {
            "observation": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "start_pos": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return PathIntegrationRecurrentActorCriticPolicy(
        obs_space,
        action_space,
        lambda _: 1e-3,
        net_arch=dict(pi=[], vf=[]),
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(drop_keys=["achieved_goal"]),
        n_lstm_layers=1,
        lstm_hidden_size=8,
        pi_bottleneck_dim=4,
        pi_n_place_cells=8,
        pi_dropout_rate=0.0,
    )


def _episode(length: int) -> dict:
    achieved_goal = np.stack(
        [np.linspace(0.0, 0.5, length, dtype=np.float32), np.linspace(0.0, 0.25, length, dtype=np.float32)],
        axis=1,
    )
    qpos = np.pad(achieved_goal, ((0, 0), (0, 1))).astype(np.float32)
    qvel = np.ones((length, 3), dtype=np.float32) * 0.1
    return {
        "obs": {
            "observation": qvel.copy(),
            "start_pos": np.repeat(achieved_goal[:1], length, axis=0),
            "achieved_goal": achieved_goal,
            "desired_goal": np.ones((length, 2), dtype=np.float32),
        },
        "action": np.zeros((length, 2), dtype=np.float32),
        "reward": np.ones((length,), dtype=np.float32),
        "terminated": np.array([False] * (length - 1) + [True], dtype=bool),
        "truncated": np.zeros((length,), dtype=bool),
        "qpos": qpos,
        "qvel": qvel,
        "goal": np.ones((length, 2), dtype=np.float32),
        "annotation": {
            "agent_xy": achieved_goal,
            "heading": np.zeros((length,), dtype=np.float32),
            "goal_xy": np.ones((length, 2), dtype=np.float32),
            "relative_goal": np.ones((length, 2), dtype=np.float32) - achieved_goal,
        },
        "summary": {
            "episode_length": length,
            "return": float(length),
            "terminated_reason": "terminated",
            "path_length": 1.0,
            "net_displacement": 1.0,
            "coverage_score": 1.0,
            "stuck_ratio": 0.0,
            "collision_ratio": 0.0,
            "goal_reached": True,
            "mean_speed": 0.1,
            "mean_turn_rate": 0.0,
        },
    }


def _write_dataset(tmp_path: Path) -> Path:
    dataset_meta = {
        "dataset_schema": POINTMAZE_MUJOCO_ZARR_SCHEMA,
        "dataset_schema_version": POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
        "dataset_name": "offline-pi-test",
    }
    write_shard(tmp_path, shard_id=0, episodes=[_episode(3), _episode(2)], dataset_meta=dataset_meta, storage_format="npz")
    return tmp_path


def _state_clone(policy: PathIntegrationRecurrentActorCriticPolicy) -> dict[str, th.Tensor]:
    return {key: value.detach().clone() for key, value in policy.state_dict().items()}


def test_online_optimizer_contains_path_integration_head_parameters():
    policy = _make_policy()
    optimizer_param_ids = {id(param) for group in policy.optimizer.param_groups for param in group["params"]}
    for param in policy.path_integration_head.parameters():
        assert id(param) in optimizer_param_ids


def test_offline_loader_and_loss_run_on_padded_recurrent_batch(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    batch = next(
        load_offline_pi_batches(
            dataset_root,
            batch_size_sequences=2,
            max_seq_len=2,
            shuffle=False,
            device="cpu",
            n_lstm_layers=1,
            lstm_hidden_size=8,
        )
    )

    assert set(batch.obs) == {"observation", "start_pos", "achieved_goal", "desired_goal"}
    assert batch.target_pos.shape == (4, 2)
    assert batch.episode_starts.shape == (4,)
    assert batch.mask.tolist() == [True, True, True, False]

    loss, metrics = compute_offline_pi_loss(policy, batch)
    assert loss.ndim == 0
    assert metrics["masked_steps"] == 3.0
    assert metrics["localization_mse"] >= 0.0
    assert metrics["localization_rmse"] >= 0.0
    assert metrics["localization_mae"] >= 0.0
    assert metrics["x_mae"] >= 0.0
    assert metrics["y_mae"] >= 0.0

    outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
    decoded = decode_offline_pi_coordinates(policy, outputs.pc_logits, batch.target_pos, batch.mask)
    assert decoded.pred_xy.shape == batch.target_pos.shape
    assert decoded.target_xy.shape == batch.target_pos.shape
    assert decoded.mask.tolist() == batch.mask.tolist()
    assert decoded.metrics["mse"] == pytest.approx(metrics["localization_mse"])


def test_count_offline_pi_sequences_respects_max_seq_len(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path)

    assert count_offline_pi_sequences(dataset_root, max_seq_len=None) == 2
    assert count_offline_pi_sequences(dataset_root, max_seq_len=2) == 3


def test_changing_only_achieved_goal_does_not_change_pi_logits():
    policy = _make_policy()
    policy.set_training_mode(False)
    obs = {
        "observation": th.zeros((2, 3), dtype=th.float32),
        "start_pos": th.zeros((2, 2), dtype=th.float32),
        "achieved_goal": th.zeros((2, 2), dtype=th.float32),
        "desired_goal": th.ones((2, 2), dtype=th.float32),
    }
    altered = {key: value.clone() for key, value in obs.items()}
    altered["achieved_goal"] = th.ones((2, 2), dtype=th.float32) * 10.0
    states = (th.zeros((1, 1, 8), dtype=th.float32), th.zeros((1, 1, 8), dtype=th.float32))
    episode_starts = th.zeros((2,), dtype=th.float32)

    with th.no_grad():
        original_outputs, _ = policy.forward_pi(obs, states, episode_starts)
        altered_outputs, _ = policy.forward_pi(altered, states, episode_starts)

    th.testing.assert_close(original_outputs.pc_logits, altered_outputs.pc_logits)


def test_offline_optimizer_membership_excludes_action_and_value_heads():
    policy = _make_policy()
    optimizer = make_offline_pi_optimizer(policy, lr=1e-3)
    offline_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    for module in (policy.pi_features_extractor, policy.lstm_actor, policy.path_integration_head):
        for param in module.parameters():
            assert id(param) in offline_param_ids
    for module in (policy.action_net, policy.value_net, policy.lstm_critic, policy.mlp_extractor):
        if module is None:
            continue
        for param in module.parameters():
            assert id(param) not in offline_param_ids


def test_probe_leaves_state_dict_unchanged(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)
    before = _state_clone(policy)

    metrics = run_offline_pi_probe(model, dataset_root, batch_size_sequences=2, max_seq_len=2)

    assert metrics["offline_pi/probe/loss"] > 0.0
    assert metrics["offline_pi/probe/localization_rmse"] >= 0.0
    assert metrics["offline_pi/probe/localization_mae"] >= 0.0
    assert metrics["offline_pi/probe/x_mae"] >= 0.0
    assert metrics["offline_pi/probe/y_mae"] >= 0.0
    for key, value in policy.state_dict().items():
        th.testing.assert_close(value, before[key])


def test_export_probe_artifacts_writes_masked_coordinate_diagnostics(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "eval"
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)

    summary = export_probe_artifacts(
        model,
        dataset_root,
        output_dir,
        epoch=0,
        batch_size_sequences=2,
        max_seq_len=2,
    )

    assert summary["num_steps"] == 5
    with np.load(output_dir / "pred_vs_target_epoch_0000.npz") as data:
        assert data["pred_xy"].shape == data["target_xy"].shape
        assert data["mask"].tolist() == [True, True, True, False, True, True]
        assert data["squared_error"].shape == data["pred_xy"].shape
        assert data["absolute_error"].shape == data["pred_xy"].shape
        assert data["bounds"].shape == (4,)
    assert (output_dir / "error_summary_epoch_0000.json").is_file()
    assert (output_dir / "coord_scatter_epoch_0000.png").is_file()
    assert (output_dir / "error_hist_epoch_0000.png").is_file()
    assert (output_dir / "spatial_error_heatmap_epoch_0000.png").is_file()


def test_export_gridscore_artifacts_writes_summary_data_and_plots(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "eval" / "gridscore_epoch_0000"
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)

    summary = export_gridscore_artifacts(
        model,
        dataset_root,
        output_dir,
        batch_size_sequences=2,
        max_seq_len=2,
        n_bins=8,
        max_steps=5,
        max_units=3,
        top_k=2,
    )

    assert summary["unit_count"] == 3
    assert summary["num_steps"] == 5
    assert summary["n_bins"] == 8
    assert summary["max_steps"] == 5
    assert summary["max_units"] == 3
    assert summary["top_k"] == 2
    with np.load(output_dir / "gridscore_data.npz") as data:
        assert data["positions"].shape == (5, 2)
        assert data["activations"].shape == (5, 3)
        assert data["ratemaps"].shape == (3, 8, 8)
        assert data["autocorrs"].shape == (3, 15, 15)
        assert data["grid_scores"].shape == (3,)
        assert data["bounds"].shape == (4,)
    assert (output_dir / "gridscore_summary.json").is_file()
    assert (output_dir / "top_grid_cells.png").is_file()
    assert (output_dir / "spatial_ratemaps_grid.png").is_file()


def test_rehearsal_rejects_policy_optimizer(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)

    with pytest.raises(ValueError, match="separate from policy.optimizer"):
        run_offline_pi_rehearsal(
            model,
            dataset_root,
            optimizer=policy.optimizer,
            batch_size_sequences=2,
            max_seq_len=2,
            max_updates=1,
        )


def test_rehearsal_updates_pi_path_but_not_action_or_value_heads(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)
    before = _state_clone(policy)
    optimizer_state_before = policy.optimizer.state_dict()

    metrics = run_offline_pi_rehearsal(
        model,
        dataset_root,
        lr=1e-2,
        batch_size_sequences=2,
        max_seq_len=2,
        max_updates=1,
        seed=0,
    )

    assert metrics["offline_pi/updates"] == 1.0
    assert metrics["offline_pi/final_epoch"] == 1.0
    assert any(
        not th.equal(policy.state_dict()[key], before[key])
        for key in before
        if key.startswith("path_integration_head.")
    )
    for key, value in policy.state_dict().items():
        if key.startswith(("action_net.", "value_net.")):
            th.testing.assert_close(value, before[key])
    assert policy.optimizer.state_dict() == optimizer_state_before
