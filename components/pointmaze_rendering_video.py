from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from components.pointmaze_rendering_data import PointMazeTrajectory, RENDER_SCHEMA_VERSION, json_safe


def localization_metrics(pred_xy: np.ndarray | None, target_xy: np.ndarray) -> dict[str, float] | None:
    if pred_xy is None:
        return None
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    if pred_xy.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    diff = pred_xy - target_xy
    per_step_mse = np.mean(diff**2, axis=-1)
    mse = float(np.mean(per_step_mse))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
    }


def frame_indices_for_length(length: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length < 0:
        raise ValueError("length must be non-negative")
    return np.arange(0, length, stride, dtype=np.int64)


def resolve_plot_bounds(
    obs_xy: np.ndarray,
    goal_xy: np.ndarray | None = None,
    pred_xy: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    if bounds is not None:
        min_x, max_x, min_y, max_y = (float(v) for v in bounds)
        if max_x <= min_x or max_y <= min_y:
            raise ValueError("bounds must satisfy min_x < max_x and min_y < max_y")
        return min_x, max_x, min_y, max_y

    parts = [np.asarray(obs_xy, dtype=np.float32).reshape(-1, 2)]
    if goal_xy is not None:
        parts.append(np.asarray(goal_xy, dtype=np.float32).reshape(-1, 2))
    if pred_xy is not None:
        parts.append(np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2))
    points = np.concatenate([part for part in parts if part.size > 0], axis=0)
    if points.size == 0:
        return (-2.5, 2.5, -2.5, 2.5)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)
    pad = np.maximum(span * 0.1, 0.25)
    return (
        float(min_xy[0] - pad[0]),
        float(max_xy[0] + pad[0]),
        float(min_xy[1] - pad[1]),
        float(max_xy[1] + pad[1]),
    )


def _xy_to_pixel(
    xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: int,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    min_x, max_x, min_y, max_y = bounds
    usable_w = max(1, width - 2 * pad - 1)
    usable_h = max(1, height - 2 * pad - 1)
    x = pad + (xy[:, 0] - min_x) / max(max_x - min_x, 1e-6) * usable_w
    y = height - pad - 1 - (xy[:, 1] - min_y) / max(max_y - min_y, 1e-6) * usable_h
    return np.stack([x, y], axis=1).round().astype(np.int32)


def _draw_line(image: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).round().astype(np.int32)
    ys = np.linspace(y0, y1, steps + 1).round().astype(np.int32)
    radius = max(0, thickness // 2)
    height, width = image.shape[:2]
    for x, y in zip(xs, ys, strict=True):
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        image[y_min:y_max, x_min:x_max] = color


def _draw_polyline(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    if points.shape[0] < 2:
        return
    for idx in range(points.shape[0] - 1):
        _draw_line(image, points[idx], points[idx + 1], color, thickness)


def _draw_disc(image: np.ndarray, point: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    x, y = int(point[0]), int(point[1])
    height, width = image.shape[:2]
    y_grid, x_grid = np.ogrid[:height, :width]
    mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius**2
    image[mask] = color


def _draw_cross(image: np.ndarray, point: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    _draw_line(image, point + np.array([-radius, 0]), point + np.array([radius, 0]), color, thickness=2)
    _draw_line(image, point + np.array([0, -radius]), point + np.array([0, radius]), color, thickness=2)


def compose_topdown_panel(
    obs_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    current_index: int,
    pred_xy: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("panel size must be positive")
    obs_xy = np.asarray(obs_xy, dtype=np.float32).reshape(-1, 2)
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        raise ValueError("obs_xy must contain at least one point")
    current_index = int(np.clip(current_index, 0, obs_xy.shape[0] - 1))
    if pred_xy is not None:
        pred_xy = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
        if pred_xy.shape[0] != obs_xy.shape[0]:
            raise ValueError("pred_xy must have the same number of rows as obs_xy")
    resolved_bounds = resolve_plot_bounds(obs_xy, goal_xy, pred_xy, bounds)

    image = np.full((height, width, 3), 248, dtype=np.uint8)
    pad = max(24, min(width, height) // 14)
    border = np.array(
        [
            [pad, pad],
            [width - pad - 1, pad],
            [width - pad - 1, height - pad - 1],
            [pad, height - pad - 1],
            [pad, pad],
        ],
        dtype=np.int32,
    )
    _draw_polyline(image, border, (205, 213, 223), thickness=2)

    path_points = _xy_to_pixel(obs_xy, resolved_bounds, width, height, pad)
    _draw_polyline(image, path_points, (148, 163, 184), thickness=1)
    _draw_polyline(image, path_points[: current_index + 1], (37, 99, 235), thickness=3)
    _draw_disc(image, path_points[0], 5, (22, 163, 74))
    _draw_disc(image, path_points[current_index], 6, (30, 64, 175))

    goal_points = _xy_to_pixel(goal_xy, resolved_bounds, width, height, pad)
    _draw_cross(image, goal_points[current_index], 7, (22, 163, 74))

    if pred_xy is not None:
        pred_points = _xy_to_pixel(pred_xy, resolved_bounds, width, height, pad)
        _draw_polyline(image, pred_points[: current_index + 1], (234, 88, 12), thickness=2)
        _draw_disc(image, pred_points[current_index], 5, (234, 88, 12))
        _draw_line(image, path_points[current_index], pred_points[current_index], (220, 38, 38), thickness=2)

    return image


def _resize_rgb_nearest(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[1] == width and image.shape[0] == height:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[y_idx][:, x_idx]


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return _resize_rgb_nearest(image, size)
    resized = cv2.resize(np.asarray(image, dtype=np.uint8), size, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def compose_video_frame(
    render_rgb: np.ndarray,
    panel_rgb: np.ndarray,
    *,
    output_size: tuple[int, int] = (960, 480),
) -> np.ndarray:
    width, height = int(output_size[0]), int(output_size[1])
    if width <= 1 or height <= 0:
        raise ValueError("output_size must be positive and at least two pixels wide")
    left_width = width // 2
    right_width = width - left_width
    left = resize_rgb(np.asarray(render_rgb, dtype=np.uint8), (left_width, height))
    right = resize_rgb(np.asarray(panel_rgb, dtype=np.uint8), (right_width, height))
    return np.concatenate([left, right], axis=1)


def _fit_mujoco_state(value: np.ndarray, template: np.ndarray | None, width: int | None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if width is None:
        return value
    if template is None:
        fitted = np.zeros((width,), dtype=np.float64)
    else:
        fitted = np.asarray(template, dtype=np.float64).reshape(-1).copy()
        if fitted.shape[0] != width:
            fitted = np.resize(fitted, width).astype(np.float64)
    copy_width = min(width, value.shape[0])
    fitted[:copy_width] = value[:copy_width]
    return fitted


def render_mujoco_state(
    env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    goal_xy: np.ndarray | None = None,
    render_size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    if goal_xy is not None and hasattr(base, "goal"):
        base.goal = np.asarray(goal_xy, dtype=np.float32)[:2].copy()
        update_target_site_pos = getattr(base, "update_target_site_pos", None)
        if callable(update_target_site_pos):
            update_target_site_pos()

    point_env = getattr(base, "point_env", base)
    model = getattr(point_env, "model", None)
    data = getattr(point_env, "data", None)
    qpos_width = int(model.nq) if model is not None and hasattr(model, "nq") else None
    qvel_width = int(model.nv) if model is not None and hasattr(model, "nv") else None
    fitted_qpos = _fit_mujoco_state(qpos, getattr(data, "qpos", None), qpos_width)
    fitted_qvel = _fit_mujoco_state(qvel, getattr(data, "qvel", None), qvel_width)
    point_env.set_state(fitted_qpos, fitted_qvel)
    frame = base.render()
    return resize_rgb(np.asarray(frame, dtype=np.uint8), render_size)


def iter_video_frames(
    trajectory: PointMazeTrajectory,
    env: Any,
    *,
    frame_indices: np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
    output_size: tuple[int, int] = (960, 480),
) -> Iterator[np.ndarray]:
    render_size = (output_size[0] // 2, output_size[1])
    panel_size = (output_size[0] - output_size[0] // 2, output_size[1])
    for idx in frame_indices:
        idx_int = int(idx)
        render_rgb = render_mujoco_state(
            env,
            trajectory.qpos[idx_int],
            trajectory.qvel[idx_int],
            goal_xy=trajectory.goal_xy[idx_int],
            render_size=render_size,
        )
        panel = compose_topdown_panel(
            trajectory.obs_xy,
            trajectory.goal_xy,
            current_index=idx_int,
            pred_xy=trajectory.pred_xy,
            bounds=bounds,
            size=panel_size,
        )
        yield compose_video_frame(render_rgb, panel, output_size=output_size)


def write_video_mp4(path: str | Path, frames: Iterable[np.ndarray], *, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_iter = iter(frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration as exc:
        raise ValueError("cannot write a video with zero frames") from exc

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    if imageio is not None:
        with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as writer:
            writer.append_data(np.asarray(first_frame, dtype=np.uint8))
            for frame in frame_iter:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("writing MP4 requires imageio or opencv-python") from exc

    first_frame = np.asarray(first_frame, dtype=np.uint8)
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    try:
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for frame in frame_iter:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _sliced(array: np.ndarray | None, frame_indices: np.ndarray) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array)[frame_indices]


def write_render_artifacts(
    trajectory: PointMazeTrajectory,
    env: Any,
    output_base: str | Path,
    *,
    fps: int = 50,
    stride: int = 1,
    bounds: tuple[float, float, float, float] | None = None,
    output_size: tuple[int, int] = (960, 480),
    command_config: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    env_substitutions: list[dict[str, str]] | None = None,
    dataset_root: str | Path | None = None,
    model_path: str | Path | None = None,
    video_writer: Callable[[str | Path, Iterable[np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_indices = frame_indices_for_length(trajectory.length, stride)
    if frame_indices.size == 0:
        raise ValueError("trajectory must contain at least one rendered frame")

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    video_path = output_base.with_suffix(".mp4")
    npz_path = output_base.with_suffix(".npz")
    json_path = output_base.with_suffix(".json")

    reset = getattr(env, "reset", None)
    if callable(reset):
        try:
            reset(seed=0)
        except TypeError:
            reset()

    frames = iter_video_frames(
        trajectory,
        env,
        frame_indices=frame_indices,
        bounds=bounds,
        output_size=output_size,
    )
    if video_writer is None:
        write_video_mp4(video_path, frames, fps=fps)
    else:
        video_writer(video_path, frames)

    pred_xy = _sliced(trajectory.pred_xy, frame_indices)
    obs_xy = _sliced(trajectory.obs_xy, frame_indices)
    error_xy = pred_xy - obs_xy if pred_xy is not None and obs_xy is not None else None
    error_norm = np.linalg.norm(error_xy, axis=1) if error_xy is not None else None
    npz_payload: dict[str, np.ndarray] = {
        "obs_xy": obs_xy,
        "qvel": _sliced(trajectory.qvel, frame_indices),
        "goal_xy": _sliced(trajectory.goal_xy, frame_indices),
        "actions": _sliced(trajectory.actions, frame_indices),
        "rewards": _sliced(trajectory.rewards, frame_indices),
        "terminated": _sliced(trajectory.terminated, frame_indices),
        "truncated": _sliced(trajectory.truncated, frame_indices),
        "post_action_qpos": _sliced(trajectory.post_action_qpos, frame_indices),
        "post_action_qvel": _sliced(trajectory.post_action_qvel, frame_indices),
        "frame_indices": frame_indices,
    }
    if pred_xy is not None:
        npz_payload["pred_xy"] = pred_xy
        npz_payload["error_xy"] = error_xy
        npz_payload["error_norm"] = error_norm
    if trajectory.bottleneck is not None:
        npz_payload["bottleneck"] = _sliced(trajectory.bottleneck, frame_indices)
    np.savez(str(npz_path), **npz_payload)

    metrics = localization_metrics(pred_xy, obs_xy) if pred_xy is not None and obs_xy is not None else None
    summary = {
        "schema_version": RENDER_SCHEMA_VERSION,
        "kind": trajectory.kind,
        "episode_id": trajectory.episode_id,
        "frame_count": int(frame_indices.shape[0]),
        "source_step_count": int(trajectory.length),
        "fps": int(fps),
        "stride": int(stride),
        "output_size": [int(output_size[0]), int(output_size[1])],
        "video_path": str(video_path),
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "model_path": None if model_path is None else str(model_path),
        "env_kwargs": env_kwargs or {},
        "env_kwargs_substitutions": env_substitutions or [],
        "command_config": command_config or {},
        "pi_metrics": metrics,
    }
    json_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
