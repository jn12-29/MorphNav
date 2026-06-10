from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from components.dataset_gen.pointmaze_config import (
    POINTMAZE_MUJOCO_ZARR_SCHEMA,
    POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION,
    POINTMAZE_POLICY_OBS_KEYS,
    default_phase1_pointmaze_xml_file_path,
    make_phase1_pointmaze_pi_env_config,
)
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs
from components.dataset_gen.pointmaze_manifest import EpisodePlan, EpisodePlanItem, load_manifest


RENDER_SCHEMA_VERSION = 1
DEFAULT_DATASET_RENDER_RUN = Path("runs/offline_pi/pointmaze_phase1_seed0")


@dataclass(frozen=True)
class EpisodeSelection:
    episode_id: int
    shard_id: int
    episode_index_in_shard: int
    shard_path: Path
    offset: int
    length: int


@dataclass(frozen=True)
class EnvKwargsResolution:
    env_kwargs: dict[str, Any]
    substitutions: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PointMazeTrajectory:
    kind: str
    episode_id: int | None
    obs: dict[str, np.ndarray]
    obs_xy: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    goal_xy: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    post_action_qpos: np.ndarray
    post_action_qvel: np.ndarray
    pred_xy: np.ndarray | None = None
    bottleneck: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.obs_xy.shape[0])

    def with_predictions(
        self,
        pred_xy: np.ndarray,
        bottleneck: np.ndarray | None = None,
    ) -> "PointMazeTrajectory":
        pred_xy = np.asarray(pred_xy, dtype=np.float32)
        if pred_xy.shape != self.obs_xy.shape:
            raise ValueError(f"pred_xy shape {pred_xy.shape} must match obs_xy shape {self.obs_xy.shape}")
        if bottleneck is not None:
            bottleneck = np.asarray(bottleneck, dtype=np.float32)
            if bottleneck.shape[0] != self.length:
                raise ValueError(
                    f"bottleneck must have {self.length} rows, got shape {bottleneck.shape}"
                )
        return replace(self, pred_xy=pred_xy, bottleneck=bottleneck)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _validate_shard_schema_values(schema: Any, version: Any, shard_path: Path) -> None:
    if schema != POINTMAZE_MUJOCO_ZARR_SCHEMA or int(version or -1) != POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION:
        raise ValueError(
            f"{shard_path} has unsupported dataset schema {schema!r} version {version!r}; "
            f"expected {POINTMAZE_MUJOCO_ZARR_SCHEMA!r} version {POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION}"
        )


def _shard_path_for_id(dataset_root: Path, shard_id: int) -> Path:
    npz_path = dataset_root / f"shard_{shard_id:06d}.npz"
    if npz_path.exists():
        return npz_path
    zarr_path = dataset_root / f"shard_{shard_id:06d}.zarr"
    if zarr_path.exists():
        return zarr_path
    raise FileNotFoundError(f"No shard file found for shard {shard_id:06d} under {dataset_root}")


def load_dataset_metadata(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "dataset_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
        return payload

    shard_paths = sorted(dataset_root.glob("shard_*.npz")) + sorted(dataset_root.glob("shard_*.zarr"))
    if not shard_paths:
        return {}
    shard_path = shard_paths[0]
    if shard_path.suffix == ".npz":
        with np.load(shard_path, allow_pickle=False) as data:
            return json.loads(str(data["dataset_meta_json"]))

    root = zarr.open_group(str(shard_path), mode="r")
    if "dataset_meta_json" not in root.attrs:
        return {}
    return json.loads(root.attrs["dataset_meta_json"])


def resolve_pointmaze_env_kwargs(dataset_root: str | Path | None = None) -> EnvKwargsResolution:
    metadata: dict[str, Any]
    if dataset_root is None:
        env_kwargs = build_pointmaze_env_kwargs(make_phase1_pointmaze_pi_env_config())
        metadata = {}
    else:
        metadata = load_dataset_metadata(dataset_root)
        raw_kwargs = metadata.get("env_kwargs")
        if raw_kwargs is None:
            env_kwargs = build_pointmaze_env_kwargs(make_phase1_pointmaze_pi_env_config())
        elif not isinstance(raw_kwargs, dict):
            raise ValueError("dataset metadata field 'env_kwargs' must be an object when present")
        else:
            env_kwargs = dict(raw_kwargs)

    substitutions: list[dict[str, str]] = []
    default_xml = default_phase1_pointmaze_xml_file_path()
    xml_file_path = env_kwargs.get("xml_file_path")
    if xml_file_path is None:
        env_kwargs["xml_file_path"] = default_xml
        substitutions.append(
            {
                "field": "xml_file_path",
                "reason": "missing",
                "replacement": default_xml,
            }
        )
    elif not Path(str(xml_file_path)).exists():
        env_kwargs["xml_file_path"] = default_xml
        substitutions.append(
            {
                "field": "xml_file_path",
                "reason": "path_not_found",
                "original": str(xml_file_path),
                "replacement": default_xml,
            }
        )

    return EnvKwargsResolution(env_kwargs=env_kwargs, substitutions=substitutions, metadata=metadata)


def _manifest_item_by_episode_id(plan: EpisodePlan, episode_id: int) -> EpisodePlanItem:
    for item in plan.episodes:
        if item.episode_id == episode_id:
            return item
    raise KeyError(f"Episode id {episode_id} is not present in manifest")


def _episode_index_in_shard(plan: EpisodePlan, item: EpisodePlanItem) -> int:
    shard_items = [candidate for candidate in plan.episodes if candidate.shard_id == item.shard_id]
    for idx, candidate in enumerate(shard_items):
        if candidate.episode_id == item.episode_id:
            return idx
    raise KeyError(f"Episode id {item.episode_id} is not present in shard {item.shard_id}")


def _load_shard_lengths_offsets(shard_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if shard_path.suffix == ".npz":
        with np.load(shard_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["dataset_meta_json"]))
            _validate_shard_schema_values(
                metadata.get("dataset_schema"),
                metadata.get("dataset_schema_version"),
                shard_path,
            )
            return (
                np.asarray(data["episode_lengths"], dtype=np.int64),
                np.asarray(data["episode_offsets"], dtype=np.int64),
            )

    root = zarr.open_group(str(shard_path), mode="r")
    _validate_shard_schema_values(root.attrs.get("dataset_schema"), root.attrs.get("dataset_schema_version"), shard_path)
    return (
        np.asarray(root["episode_lengths"][:], dtype=np.int64),
        np.asarray(root["episode_offsets"][:], dtype=np.int64),
    )


def resolve_episode_selection(dataset_root: str | Path, episode_id: int) -> EpisodeSelection:
    dataset_root = Path(dataset_root)
    plan = load_manifest(dataset_root / "manifest.json")
    item = _manifest_item_by_episode_id(plan, int(episode_id))
    episode_index = _episode_index_in_shard(plan, item)
    shard_path = _shard_path_for_id(dataset_root, item.shard_id)
    lengths, offsets = _load_shard_lengths_offsets(shard_path)
    if episode_index >= int(lengths.shape[0]):
        raise IndexError(
            f"Manifest maps episode {episode_id} to shard-local index {episode_index}, "
            f"but {shard_path} contains only {lengths.shape[0]} episodes"
        )
    return EpisodeSelection(
        episode_id=int(item.episode_id),
        shard_id=int(item.shard_id),
        episode_index_in_shard=int(episode_index),
        shard_path=shard_path,
        offset=int(offsets[episode_index]),
        length=int(lengths[episode_index]),
    )


def _slice_arrays(arrays: dict[str, np.ndarray], selection: EpisodeSelection) -> dict[str, np.ndarray]:
    start = selection.offset
    end = selection.offset + selection.length
    return {key: np.asarray(value[start:end]) for key, value in arrays.items()}


def _load_npz_episode_arrays(shard_path: Path, selection: EpisodeSelection) -> dict[str, np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["dataset_meta_json"]))
        _validate_shard_schema_values(
            metadata.get("dataset_schema"),
            metadata.get("dataset_schema_version"),
            shard_path,
        )
        arrays = {
            "actions": np.asarray(data["step/action"], dtype=np.float32),
            "rewards": np.asarray(data["step/reward"], dtype=np.float32),
            "terminated": np.asarray(data["step/terminated"], dtype=bool),
            "truncated": np.asarray(data["step/truncated"], dtype=bool),
            "post_action_qpos": np.asarray(data["step/qpos"], dtype=np.float32),
            "post_action_qvel": np.asarray(data["step/qvel"], dtype=np.float32),
            "step_goal": np.asarray(data["step/goal"], dtype=np.float32),
        }
        for key in POINTMAZE_POLICY_OBS_KEYS:
            arrays[f"obs/{key}"] = np.asarray(data[f"obs/{key}"], dtype=np.float32)
    return _slice_arrays(arrays, selection)


def _load_zarr_episode_arrays(shard_path: Path, selection: EpisodeSelection) -> dict[str, np.ndarray]:
    root = zarr.open_group(str(shard_path), mode="r")
    _validate_shard_schema_values(root.attrs.get("dataset_schema"), root.attrs.get("dataset_schema_version"), shard_path)
    arrays = {
        "actions": np.asarray(root["step/action"][:], dtype=np.float32),
        "rewards": np.asarray(root["step/reward"][:], dtype=np.float32),
        "terminated": np.asarray(root["step/terminated"][:], dtype=bool),
        "truncated": np.asarray(root["step/truncated"][:], dtype=bool),
        "post_action_qpos": np.asarray(root["step/qpos"][:], dtype=np.float32),
        "post_action_qvel": np.asarray(root["step/qvel"][:], dtype=np.float32),
        "step_goal": np.asarray(root["step/goal"][:], dtype=np.float32),
    }
    for key in POINTMAZE_POLICY_OBS_KEYS:
        arrays[f"obs/{key}"] = np.asarray(root[f"obs/{key}"][:], dtype=np.float32)
    return _slice_arrays(arrays, selection)


def _action_before_qpos(obs_xy: np.ndarray, post_action_qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(post_action_qpos, dtype=np.float32).copy()
    if qpos.ndim != 2 or qpos.shape[0] != obs_xy.shape[0] or qpos.shape[1] < 2:
        raise ValueError(f"step/qpos must have shape (T, >=2), got {qpos.shape}")
    qpos[:, :2] = obs_xy[:, :2]
    return qpos


def _action_before_qvel(obs_observation: np.ndarray, post_action_qvel: np.ndarray) -> np.ndarray:
    qvel = np.asarray(post_action_qvel, dtype=np.float32).copy()
    obs_observation = np.asarray(obs_observation, dtype=np.float32)
    if qvel.ndim != 2:
        raise ValueError(f"step/qvel must have shape (T, D), got {qvel.shape}")
    if obs_observation.ndim != 2 or obs_observation.shape[0] != qvel.shape[0]:
        return qvel
    width = min(obs_observation.shape[1], qvel.shape[1])
    qvel[:, :width] = obs_observation[:, :width]
    return qvel


def load_dataset_episode(dataset_root: str | Path, episode_id: int, *, kind: str = "dataset") -> PointMazeTrajectory:
    selection = resolve_episode_selection(dataset_root, episode_id)
    if selection.shard_path.suffix == ".npz":
        arrays = _load_npz_episode_arrays(selection.shard_path, selection)
    else:
        arrays = _load_zarr_episode_arrays(selection.shard_path, selection)

    obs = {key: np.asarray(arrays[f"obs/{key}"], dtype=np.float32) for key in POINTMAZE_POLICY_OBS_KEYS}
    obs_xy = np.asarray(obs["achieved_goal"], dtype=np.float32)[..., :2]
    goal_xy = np.asarray(obs.get("desired_goal", arrays["step_goal"]), dtype=np.float32)[..., :2]
    post_qpos = np.asarray(arrays["post_action_qpos"], dtype=np.float32)
    post_qvel = np.asarray(arrays["post_action_qvel"], dtype=np.float32)
    qpos = _action_before_qpos(obs_xy, post_qpos)
    qvel = _action_before_qvel(obs["observation"], post_qvel)

    return PointMazeTrajectory(
        kind=kind,
        episode_id=int(episode_id),
        obs=obs,
        obs_xy=obs_xy,
        qpos=qpos,
        qvel=qvel,
        goal_xy=goal_xy,
        actions=np.asarray(arrays["actions"], dtype=np.float32),
        rewards=np.asarray(arrays["rewards"], dtype=np.float32),
        terminated=np.asarray(arrays["terminated"], dtype=bool),
        truncated=np.asarray(arrays["truncated"], dtype=bool),
        post_action_qpos=post_qpos,
        post_action_qvel=post_qvel,
    )


def offline_pi_run_root_from_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.parent.parent if path.parent.name == "models" else path.parent


def resolve_render_output_dir(
    *,
    mode: str,
    dataset_root: str | Path | None,
    model_path: str | Path | None,
    output_dir: str | Path | None,
    seed: int = 0,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if mode == "dataset":
        if dataset_root is None:
            raise ValueError("dataset mode requires dataset_root")
        return DEFAULT_DATASET_RENDER_RUN / "analysis" / "renders" / Path(dataset_root).name
    if mode == "probe":
        if dataset_root is None or model_path is None:
            raise ValueError("probe mode requires dataset_root and model_path")
        run_root = offline_pi_run_root_from_model_path(model_path)
        return run_root / "analysis" / "renders" / Path(dataset_root).name
    if mode == "rollout":
        if model_path is None:
            raise ValueError("rollout mode requires model_path")
        run_root = offline_pi_run_root_from_model_path(model_path)
        return run_root / "analysis" / "renders" / f"rollout_seed{seed}"
    raise ValueError(f"unsupported render mode {mode!r}")
