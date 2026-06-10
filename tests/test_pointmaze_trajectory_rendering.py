import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
    default_phase1_pointmaze_xml_file_path,
)
from components.dataset_gen.pointmaze_manifest import EpisodePlan, EpisodePlanItem, save_manifest
from components.dataset_gen.pointmaze_zarr_writer import write_dataset_metadata, write_shard
from components.pointmaze_trajectory_rendering import (
    compose_topdown_panel,
    load_dataset_episode,
    predict_dataset_episode_pi,
    resolve_episode_selection,
    resolve_pointmaze_env_kwargs,
    resolve_render_output_dir,
    write_render_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "render_pointmaze_trajectory.py"
SPEC = importlib.util.spec_from_file_location("render_pointmaze_trajectory_script", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load render script at {SCRIPT_PATH}")
render_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render_script
SPEC.loader.exec_module(render_script)


def _episode(obs_xy: np.ndarray, *, post_offset: float = 10.0) -> dict:
    obs_xy = np.asarray(obs_xy, dtype=np.float32)
    length = int(obs_xy.shape[0])
    post_qpos = np.pad(obs_xy + post_offset, ((0, 0), (0, 1))).astype(np.float32)
    post_qvel = np.ones((length, 3), dtype=np.float32) * 3.0
    obs_observation = np.column_stack(
        [
            np.linspace(0.0, 0.2, length, dtype=np.float32),
            np.linspace(0.3, 0.5, length, dtype=np.float32),
            np.linspace(0.6, 0.8, length, dtype=np.float32),
        ]
    )
    goal = np.repeat(np.asarray([[2.0, 2.5]], dtype=np.float32), length, axis=0)
    return {
        "obs": {
            "observation": obs_observation,
            "start_pos": np.repeat(obs_xy[:1], length, axis=0),
            "achieved_goal": obs_xy,
            "desired_goal": goal,
        },
        "action": np.zeros((length, 2), dtype=np.float32),
        "reward": np.ones((length,), dtype=np.float32),
        "terminated": np.array([False] * (length - 1) + [True], dtype=bool),
        "truncated": np.zeros((length,), dtype=bool),
        "qpos": post_qpos,
        "qvel": post_qvel,
        "goal": goal,
        "annotation": {
            "agent_xy": obs_xy,
            "heading": np.zeros((length,), dtype=np.float32),
            "goal_xy": goal,
            "relative_goal": goal - obs_xy,
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


def _write_dataset(root: Path, *, stale_xml: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_schema": POINTMAZE_MUJOCO_ZARR_SCHEMA,
        "dataset_schema_version": POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
        "dataset_name": "render-test",
        "env_kwargs": {
            "maze_map_name": "OPEN",
            "continuing_task": False,
            "reset_target": True,
            "sensor_aware": True,
            "achieved_goal_aware": True,
            "start_pos_aware": True,
            "target_aware": True,
            "success_radius": 0.4,
            "xml_file_path": "/missing/point_v1.xml" if stale_xml else default_phase1_pointmaze_xml_file_path(),
        },
    }
    write_shard(
        root,
        shard_id=0,
        episodes=[
            _episode(np.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]], dtype=np.float32)),
            _episode(np.asarray([[1.0, 0.0], [1.2, 0.2]], dtype=np.float32)),
        ],
        dataset_meta=metadata,
        storage_format="npz",
    )
    write_shard(
        root,
        shard_id=1,
        episodes=[_episode(np.asarray([[-1.0, 0.0], [-0.8, 0.2]], dtype=np.float32))],
        dataset_meta=metadata,
        storage_format="npz",
    )
    save_manifest(
        EpisodePlan(
            dataset_seed=0,
            episodes=[
                EpisodePlanItem(episode_id=0, shard_id=0, episode_seed=11),
                EpisodePlanItem(episode_id=1, shard_id=0, episode_seed=22),
                EpisodePlanItem(episode_id=2, shard_id=1, episode_seed=33),
            ],
        ),
        root / "manifest.json",
    )
    write_dataset_metadata(root, metadata)
    return root


def test_resolve_episode_selection_uses_manifest_shard_order(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")

    selection = resolve_episode_selection(dataset_root, 2)

    assert selection.shard_id == 1
    assert selection.episode_index_in_shard == 0
    assert selection.offset == 0
    assert selection.length == 2
    assert selection.shard_path.name == "shard_000001.npz"


def test_load_dataset_episode_uses_action_before_observation_state(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")

    trajectory = load_dataset_episode(dataset_root, 0)

    np.testing.assert_allclose(trajectory.qpos[:, :2], trajectory.obs_xy)
    assert np.all(trajectory.post_action_qpos[:, :2] > trajectory.obs_xy + 9.0)
    np.testing.assert_allclose(trajectory.qvel[:, :3], trajectory.obs["observation"])


def test_resolve_env_kwargs_replaces_stale_xml_path(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset", stale_xml=True)

    resolution = resolve_pointmaze_env_kwargs(dataset_root)

    assert resolution.env_kwargs["xml_file_path"] == default_phase1_pointmaze_xml_file_path()
    assert resolution.substitutions == [
        {
            "field": "xml_file_path",
            "reason": "path_not_found",
            "original": "/missing/point_v1.xml",
            "replacement": default_phase1_pointmaze_xml_file_path(),
        }
    ]


def test_render_output_dir_defaults_match_modes():
    dataset_dir = Path("data/datasets/pointmaze/phase1_pi/probe_seed1")
    model_path = Path("runs/offline_pi/run/models/final_model.zip")

    assert resolve_render_output_dir(mode="dataset", dataset_root=dataset_dir, model_path=None, output_dir=None) == Path(
        "runs/offline_pi/pointmaze_phase1_seed0/analysis/renders/probe_seed1"
    )
    assert resolve_render_output_dir(mode="probe", dataset_root=dataset_dir, model_path=model_path, output_dir=None) == Path(
        "runs/offline_pi/run/analysis/renders/probe_seed1"
    )
    assert resolve_render_output_dir(
        mode="rollout",
        dataset_root=None,
        model_path=model_path,
        output_dir=None,
        seed=7,
    ) == Path("runs/offline_pi/run/analysis/renders/rollout_seed7")


def test_render_script_cli_validation_requires_mode_specific_inputs():
    parser = render_script.build_parser()
    args = parser.parse_args(["--mode", "dataset"])
    with pytest.raises(SystemExit):
        render_script._validate_args(parser, args)

    args = parser.parse_args(["--mode", "dataset", "--dataset-root", "data/datasets/pointmaze/phase1_pi/rehearsal_seed0"])
    with pytest.raises(SystemExit):
        render_script._validate_args(parser, args)

    args = parser.parse_args(
        [
            "--mode",
            "probe",
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/probe_seed1",
            "--model-path",
            "runs/offline_pi/run/models/final_model.zip",
        ]
    )
    with pytest.raises(SystemExit):
        render_script._validate_args(parser, args)

    args = parser.parse_args(
        [
            "--mode",
            "probe",
            "--dataset-root",
            "data/datasets/pointmaze/phase1_pi/probe_seed1",
            "--episodes",
            "0",
        ]
    )
    with pytest.raises(SystemExit):
        render_script._validate_args(parser, args)

    args = parser.parse_args(["--mode", "rollout", "--model-path", "runs/offline_pi/run/models/final_model.zip"])
    render_script._validate_args(parser, args)


def test_topdown_panel_draws_prediction_and_error_overlay():
    obs_xy = np.asarray([[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]], dtype=np.float32)
    pred_xy = obs_xy + np.asarray([0.1, -0.1], dtype=np.float32)
    goal_xy = np.repeat(np.asarray([[1.5, 1.5]], dtype=np.float32), 3, axis=0)

    panel = compose_topdown_panel(obs_xy, goal_xy, current_index=2, pred_xy=pred_xy, size=(160, 120))

    assert panel.shape == (120, 160, 3)
    assert panel.dtype == np.uint8
    assert np.count_nonzero(np.all(panel == np.asarray([234, 88, 12], dtype=np.uint8), axis=2)) > 0
    assert np.count_nonzero(np.all(panel == np.asarray([220, 38, 38], dtype=np.uint8), axis=2)) > 0


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
        self.device = th.device("cpu")
        self.lstm_hidden_state_shape = (1, 1, 1)
        self.path_integration_target_encoder = _FakeTargetEncoder()
        self.start_pos_calls: list[np.ndarray] = []

    def set_training_mode(self, mode: bool) -> None:
        self.training = bool(mode)

    def forward_pi(self, obs, states, episode_starts):
        target = obs["achieved_goal"].detach().cpu().numpy()
        start_pos = obs["start_pos"].detach().cpu().numpy()
        self.start_pos_calls.append(start_pos.copy())
        centers = self.path_integration_target_encoder.centers.detach().cpu().numpy()
        logits = -np.sum((target[:, None, :] - centers[None, :, :]) ** 2, axis=-1) * 16.0
        bottleneck = np.column_stack([target[:, 0], target[:, 1], target[:, 0] + target[:, 1]])
        return (
            SimpleNamespace(
                pc_logits=th.as_tensor(logits, dtype=th.float32),
                bottleneck=th.as_tensor(bottleneck, dtype=th.float32),
            ),
            states,
        )


def test_probe_prediction_chunks_align_start_pos_to_action_before_observation(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")
    trajectory = load_dataset_episode(dataset_root, 0, kind="probe")
    policy = _FakePolicy()
    model = SimpleNamespace(policy=policy)

    predicted = predict_dataset_episode_pi(model, trajectory, max_seq_len=2)

    assert predicted.pred_xy.shape == trajectory.obs_xy.shape
    assert predicted.bottleneck.shape == (trajectory.length, 3)
    np.testing.assert_allclose(policy.start_pos_calls[0], np.repeat(trajectory.obs_xy[:1], 2, axis=0))
    np.testing.assert_allclose(policy.start_pos_calls[1], trajectory.obs_xy[2:3])
    assert policy.training is True


class _FakeRenderEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.point_env = self
        self.model = SimpleNamespace(nq=2, nv=2)
        self.data = SimpleNamespace(qpos=np.zeros((2,), dtype=np.float32), qvel=np.zeros((2,), dtype=np.float32))
        self.goal = np.zeros((2,), dtype=np.float32)
        self.states: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self, seed=None):
        return None

    def update_target_site_pos(self):
        return None

    def set_state(self, qpos, qvel):
        self.states.append((np.asarray(qpos).copy(), np.asarray(qvel).copy()))

    def render(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[..., 0] = 40
        frame[..., 1] = 80
        frame[..., 2] = 120
        return frame


def test_write_render_artifacts_writes_npz_json_and_uses_selected_frames(tmp_path: Path):
    dataset_root = _write_dataset(tmp_path / "dataset")
    trajectory = load_dataset_episode(dataset_root, 0, kind="probe").with_predictions(
        np.asarray([[0.1, 0.0], [0.3, 0.2], [0.6, 0.3]], dtype=np.float32)
    )
    env = _FakeRenderEnv()
    written: dict[str, int] = {}

    def fake_video_writer(path, frames):
        count = 0
        for frame in frames:
            assert frame.shape == (64, 128, 3)
            count += 1
        Path(path).write_bytes(b"fake mp4")
        written["count"] = count

    summary = write_render_artifacts(
        trajectory,
        env,
        tmp_path / "render" / "episode_000000",
        fps=10,
        stride=2,
        output_size=(128, 64),
        command_config={"mode": "probe"},
        env_kwargs={"maze_map_name": "OPEN"},
        env_substitutions=[{"field": "xml_file_path", "reason": "missing", "replacement": "x"}],
        dataset_root=dataset_root,
        model_path="runs/offline_pi/run/models/final_model.zip",
        video_writer=fake_video_writer,
    )

    assert written["count"] == 2
    assert all(qpos.shape == (2,) and qvel.shape == (2,) for qpos, qvel in env.states)
    assert (tmp_path / "render" / "episode_000000.mp4").read_bytes() == b"fake mp4"
    with np.load(tmp_path / "render" / "episode_000000.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_indices"], np.asarray([0, 2], dtype=np.int64))
        assert data["obs_xy"].shape == (2, 2)
        assert data["pred_xy"].shape == (2, 2)
        assert data["error_norm"].shape == (2,)
        np.testing.assert_allclose(data["post_action_qpos"][:, :2], trajectory.post_action_qpos[[0, 2], :2])
    payload = json.loads((tmp_path / "render" / "episode_000000.json").read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["pi_metrics"]["mse"] >= 0.0
    assert payload["env_kwargs_substitutions"][0]["reason"] == "missing"
    assert summary["npz_path"].endswith("episode_000000.npz")


@pytest.mark.skipif(
    os.environ.get("MORPHNAV_RUN_MUJOCO_RENDER_SMOKE") != "1",
    reason="set MORPHNAV_RUN_MUJOCO_RENDER_SMOKE=1 and MUJOCO_GL=egl to run the MuJoCo MP4 smoke test",
)
def test_mujoco_dataset_render_smoke_writes_nonempty_artifacts(tmp_path: Path):
    import gymnasium as gym

    import envs  # noqa: F401

    dataset_root = _write_dataset(tmp_path / "dataset")
    resolution = resolve_pointmaze_env_kwargs(dataset_root)
    kwargs = dict(resolution.env_kwargs)
    kwargs["render_mode"] = "rgb_array"
    env = gym.make("PointMaze", **kwargs)
    try:
        trajectory = load_dataset_episode(dataset_root, 0)
        summary = write_render_artifacts(
            trajectory,
            env,
            tmp_path / "smoke" / "episode_000000",
            fps=5,
            stride=1,
            output_size=(160, 80),
            env_kwargs=resolution.env_kwargs,
            env_substitutions=resolution.substitutions,
            dataset_root=dataset_root,
        )
    finally:
        env.close()

    assert Path(summary["video_path"]).stat().st_size > 0
    assert Path(summary["npz_path"]).stat().st_size > 0
    assert Path(summary["json_path"]).stat().st_size > 0
