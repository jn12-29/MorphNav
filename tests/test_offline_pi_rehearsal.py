import inspect
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
import components.offline_pi_gridscore as gridscore_module
from components.offline_pi_gridscore import export_gridscore_artifacts
from components.offline_pi_rehearsal import (
    _mean_metrics,
    count_offline_pi_sequences,
    compute_offline_pi_loss,
    decode_offline_pi_coordinates,
    first_step_mask_from_batch,
    load_offline_pi_batches,
    make_offline_pi_optimizer,
    resolve_offline_pi_optimizer_class,
    run_offline_pi_probe,
    run_offline_pi_rehearsal,
)
from components.path_integration import recurrent_first_step_loss_weights, soft_place_cell_cross_entropy
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
        pi_init_state_key="start_pos",
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


def test_online_optimizer_contains_path_integration_parameters():
    policy = _make_policy()
    optimizer_param_ids = {id(param) for group in policy.optimizer.param_groups for param in group["params"]}
    for module in policy._path_integration_trainable_modules():
        if module is None:
            continue
        for param in module.parameters():
            assert id(param) in optimizer_param_ids


def test_soft_place_cell_cross_entropy_supports_first_step_weights():
    pc_logits = th.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    pc_targets = th.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    mask = th.tensor([True, True, True, False])

    unit_weights = recurrent_first_step_loss_weights(mask, sequence_count=2, first_step_weight=1.0)
    weighted = recurrent_first_step_loss_weights(mask, sequence_count=2, first_step_weight=10.0)
    per_step = -(pc_targets * th.log_softmax(pc_logits, dim=-1)).sum(dim=-1)

    unweighted_loss = soft_place_cell_cross_entropy(pc_logits, pc_targets, mask=mask)
    unit_weight_loss = soft_place_cell_cross_entropy(pc_logits, pc_targets, mask=mask, weights=unit_weights)
    weighted_loss = soft_place_cell_cross_entropy(pc_logits, pc_targets, mask=mask, weights=weighted)

    th.testing.assert_close(unit_weight_loss, unweighted_loss)
    th.testing.assert_close(weighted, th.tensor([10.0, 1.0, 10.0, 1.0]))
    expected = (per_step[0] * 10.0 + per_step[1] + per_step[2] * 10.0) / 21.0
    th.testing.assert_close(weighted_loss, expected)


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
    assert first_step_mask_from_batch(batch).tolist() == [True, False, True, False]
    np.testing.assert_allclose(batch.obs["start_pos"][2].cpu().numpy(), batch.target_pos[2].cpu().numpy())

    loss, metrics = compute_offline_pi_loss(policy, batch, first_step_loss_weight=10.0)
    assert loss.ndim == 0
    assert metrics["loss_weight_sum"] == 21.0
    assert metrics["masked_steps"] == 3.0
    assert metrics["first_step_count"] == 2.0
    assert metrics["localization_mse"] >= 0.0
    assert metrics["localization_rmse"] >= 0.0
    assert metrics["localization_mae"] >= 0.0
    assert metrics["x_mae"] >= 0.0
    assert metrics["y_mae"] >= 0.0
    assert metrics["first_localization_mse"] >= 0.0
    assert metrics["first_localization_rmse"] >= 0.0
    assert metrics["first_localization_mae"] >= 0.0
    assert metrics["first_x_mae"] >= 0.0
    assert metrics["first_y_mae"] >= 0.0
    assert metrics["first_localization_mse_ratio"] >= 0.0
    assert metrics["first_localization_mae_ratio"] >= 0.0

    outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
    decoded = decode_offline_pi_coordinates(policy, outputs.pc_logits, batch.target_pos, batch.mask)
    assert decoded.pred_xy.shape == batch.target_pos.shape
    assert decoded.target_xy.shape == batch.target_pos.shape
    assert decoded.mask.tolist() == batch.mask.tolist()
    assert decoded.metrics["mse"] == pytest.approx(metrics["localization_mse"])


def test_compute_offline_pi_loss_weights_sequence_first_steps(tmp_path: Path):
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

    default_loss, default_metrics = compute_offline_pi_loss(policy, batch)
    unweighted_loss, unweighted_metrics = compute_offline_pi_loss(policy, batch, first_step_loss_weight=1.0)
    weighted_loss, weighted_metrics = compute_offline_pi_loss(policy, batch, first_step_loss_weight=10.0)
    outputs, _ = policy.forward_pi(batch.obs, batch.lstm_states_pi, batch.episode_starts)
    targets = policy.path_integration_target_encoder(batch.target_pos).to(dtype=outputs.pc_logits.dtype)
    expected = soft_place_cell_cross_entropy(
        outputs.pc_logits,
        targets,
        mask=batch.mask,
        weights=recurrent_first_step_loss_weights(
            batch.mask,
            sequence_count=batch.sequence_count,
            first_step_weight=10.0,
        ),
    )

    th.testing.assert_close(default_loss, weighted_loss)
    th.testing.assert_close(weighted_loss, expected)
    assert default_metrics["loss_weight_sum"] == pytest.approx(weighted_metrics["loss_weight_sum"])
    assert float(weighted_loss.detach().cpu().item()) != pytest.approx(
        float(unweighted_loss.detach().cpu().item())
    )
    assert weighted_metrics["localization_mse"] == pytest.approx(unweighted_metrics["localization_mse"])


def _metric_row(
    *,
    loss: float,
    loss_weight_sum: float,
    masked_steps: float,
    first_step_count: float,
) -> dict[str, float]:
    return {
        "loss": loss,
        "loss_weight_sum": loss_weight_sum,
        "localization_mse": 4.0,
        "localization_mae": 2.0,
        "x_mae": 1.0,
        "y_mae": 3.0,
        "first_localization_mse": 1.0,
        "first_localization_mae": 1.0,
        "first_x_mae": 1.0,
        "first_y_mae": 1.0,
        "first_localization_mse_ratio": 0.25,
        "first_localization_mae_ratio": 0.5,
        "masked_steps": masked_steps,
        "first_step_count": first_step_count,
        "sequence_count": 1.0,
    }


def test_mean_metrics_aggregates_weighted_loss_by_effective_loss_weights():
    metrics = _mean_metrics(
        [
            _metric_row(loss=10.0, loss_weight_sum=19.0, masked_steps=10.0, first_step_count=1.0),
            _metric_row(loss=1.0, loss_weight_sum=10.0, masked_steps=1.0, first_step_count=1.0),
        ],
        "offline_pi",
    )

    assert metrics["offline_pi/loss"] == pytest.approx((10.0 * 19.0 + 1.0 * 10.0) / 29.0)
    assert metrics["offline_pi/loss_weight_sum"] == 29.0
    assert metrics["offline_pi/steps"] == 11.0


def test_offline_pi_public_helpers_default_to_first_step_weight_10():
    assert inspect.signature(compute_offline_pi_loss).parameters["first_step_loss_weight"].default == 10.0
    assert inspect.signature(run_offline_pi_rehearsal).parameters["first_step_loss_weight"].default == 10.0
    assert inspect.signature(run_offline_pi_probe).parameters["first_step_loss_weight"].default == 10.0


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


def test_start_pos_initializes_lstm_state_at_episode_start():
    policy = _make_policy()
    policy.set_training_mode(False)
    init_positions = policy.path_integration_target_encoder.centers[:2].detach().cpu()
    obs = {
        "observation": th.zeros((2, 3), dtype=th.float32),
        "start_pos": init_positions.to(dtype=th.float32),
        "achieved_goal": th.zeros((2, 2), dtype=th.float32),
        "desired_goal": th.ones((2, 2), dtype=th.float32),
    }
    states = (th.zeros((1, 2, 8), dtype=th.float32), th.zeros((1, 2, 8), dtype=th.float32))
    episode_starts = th.ones((2,), dtype=th.float32)

    with th.no_grad():
        outputs, _ = policy.forward_pi(obs, states, episode_starts)

    assert not th.allclose(outputs.pc_logits[0], outputs.pc_logits[1])


def test_offline_optimizer_membership_excludes_action_and_value_heads():
    policy = _make_policy()
    optimizer = make_offline_pi_optimizer(policy, lr=1e-3)
    offline_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    for module in (
        policy.pi_features_extractor,
        policy.lstm_actor,
        policy.path_integration_state_init,
        policy.path_integration_cell_init,
        policy.path_integration_head,
    ):
        for param in module.parameters():
            assert id(param) in offline_param_ids
    for module in (policy.action_net, policy.value_net, policy.lstm_critic, policy.mlp_extractor):
        if module is None:
            continue
        for param in module.parameters():
            assert id(param) not in offline_param_ids


def test_offline_optimizer_factory_supports_configured_optimizer_kwargs():
    policy = _make_policy()

    optimizer = make_offline_pi_optimizer(
        policy,
        lr=1e-3,
        optimizer_cls=resolve_offline_pi_optimizer_class("sgd"),
        weight_decay=0.02,
        momentum=0.9,
    )

    assert isinstance(optimizer, th.optim.SGD)
    assert optimizer.defaults["weight_decay"] == 0.02
    assert optimizer.defaults["momentum"] == 0.9


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
    assert metrics["offline_pi/probe/first_step_count"] == 3.0
    assert metrics["offline_pi/probe/first_localization_mse"] >= 0.0
    assert metrics["offline_pi/probe/first_localization_rmse"] >= 0.0
    assert metrics["offline_pi/probe/first_localization_mae"] >= 0.0
    assert metrics["offline_pi/probe/first_localization_mse_ratio"] >= 0.0
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
    assert summary["first_step"]["num_steps"] == 3
    assert summary["first_step"]["mse"] >= 0.0
    assert summary["first_step"]["mse_ratio"] >= 0.0
    with np.load(output_dir / "pred_vs_target_epoch_0000.npz") as data:
        assert data["pred_xy"].shape == data["target_xy"].shape
        assert data["mask"].tolist() == [True, True, True, False, True, True]
        assert data["first_step_mask"].tolist() == [True, False, True, False, True, False]
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
    assert summary["gridscore_positive_activations"] is False
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


def test_gridscore_positive_activations_clips_analysis_weights_and_preserves_raw(monkeypatch):
    positions = np.asarray(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [1.5, 1.5],
            [1.5, 1.5],
        ],
        dtype=np.float32,
    )
    activations = np.asarray(
        [
            [-2.0, 1.0],
            [4.0, -3.0],
            [-5.0, -6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        gridscore_module,
        "collect_bottleneck_activity",
        lambda *args, **kwargs: (positions, activations),
    )

    analysis = gridscore_module.compute_gridscore_analysis(
        object(),
        Path("unused"),
        batch_size_sequences=1,
        max_seq_len=1,
        n_bins=2,
        bounds=(0.0, 2.0, 0.0, 2.0),
        max_units=None,
        gridscore_positive_activations=True,
    )

    np.testing.assert_array_equal(analysis["activations"], activations)
    assert analysis["summary"]["gridscore_positive_activations"] is True
    assert analysis["ratemaps"][0, 0, 0] == pytest.approx(2.0)
    assert analysis["ratemaps"][0, 1, 1] == pytest.approx(3.5)
    assert analysis["ratemaps"][1, 0, 0] == pytest.approx(0.5)
    assert analysis["ratemaps"][1, 1, 1] == pytest.approx(4.0)


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


def test_rehearsal_rejects_negative_max_grad_norm(tmp_path: Path):
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)

    with pytest.raises(ValueError, match="max_grad_norm must be non-negative"):
        run_offline_pi_rehearsal(
            model,
            tmp_path,
            max_grad_norm=-0.1,
        )


def test_rehearsal_clips_offline_optimizer_params_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)
    optimizer = make_offline_pi_optimizer(policy, lr=1e-2)
    expected_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    policy_param_ids = {id(param) for param in policy.parameters()}
    events: list[str] = []
    calls: list[tuple[list[th.nn.Parameter], float]] = []
    real_step = optimizer.step

    def fake_clip_grad_norm_(params, max_norm):
        params = list(params)
        assert any(param.grad is not None for param in params)
        events.append("clip")
        calls.append((params, max_norm))
        return th.tensor(0.0)

    def fake_step(*args, **kwargs):
        assert events == ["clip"]
        events.append("step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(th.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)
    monkeypatch.setattr(optimizer, "step", fake_step)

    run_offline_pi_rehearsal(
        model,
        dataset_root,
        optimizer=optimizer,
        batch_size_sequences=2,
        max_seq_len=2,
        max_updates=1,
        seed=0,
    )

    assert len(calls) == 1
    assert events == ["clip", "step"]
    clipped_params, max_norm = calls[0]
    assert max_norm == 0.5
    assert {id(param) for param in clipped_params} == expected_param_ids
    assert expected_param_ids < policy_param_ids


def test_rehearsal_max_grad_norm_zero_disables_clipping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_root = _write_dataset(tmp_path)
    policy = _make_policy()
    model = SimpleNamespace(policy=policy)
    calls = 0

    def fake_clip_grad_norm_(params, max_norm):
        nonlocal calls
        calls += 1
        return th.tensor(0.0)

    monkeypatch.setattr(th.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    run_offline_pi_rehearsal(
        model,
        dataset_root,
        batch_size_sequences=2,
        max_seq_len=2,
        max_updates=1,
        seed=0,
        max_grad_norm=0.0,
    )

    assert calls == 0


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
    assert metrics["offline_pi/first_step_count"] > 0.0
    assert metrics["offline_pi/first_localization_mse"] >= 0.0
    assert metrics["offline_pi/first_localization_mse_ratio"] >= 0.0
    assert any(
        not th.equal(policy.state_dict()[key], before[key])
        for key in before
        if key.startswith((
            "path_integration_head.",
            "path_integration_state_init.",
            "path_integration_cell_init.",
        ))
    )
    for key, value in policy.state_dict().items():
        if key.startswith(("action_net.", "value_net.")):
            th.testing.assert_close(value, before[key])
    assert policy.optimizer.state_dict() == optimizer_state_before
