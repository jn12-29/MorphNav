"""PointMaze dataset schema constants and Phase 1 preset.

The Phase 1 preset owns the dataset-generation environment defaults that must
match the target `pi_ppo_lstm` run: sensor-aware PointMaze with `achieved_goal`,
`start_pos`, and `desired_goal` present in observations. Compact NPZ shards are
the default storage format.
"""

from dataclasses import dataclass, field
from pathlib import Path


POINTMAZE_MUJOCO_ZARR_SCHEMA = "pointmaze_mujoco_zarr"
POINTMAZE_MUJOCO_ZARR_SCHEMA_VERSION = 1
POINTMAZE_POLICY_OBS_KEYS = ("observation", "start_pos", "achieved_goal", "desired_goal")
PHASE1_POINTMAZE_PI_PRESET = "phase1_pointmaze_pi"


def default_phase1_pointmaze_xml_file_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "envs" / "assets" / "point_v1.xml")


@dataclass
class PointMazeEnvConfig:
    env_id: str = "PointMaze"
    maze_map_name: str = "OPEN"
    xml_file_path: str = default_phase1_pointmaze_xml_file_path()
    continuing_task: bool = True
    reset_target: bool = False
    # Collection-side safety cap; applied by dataset collector, not env constructor kwargs.
    max_episode_steps: int = 1000
    sensor_aware: bool = True
    achieved_goal_aware: bool = True
    start_pos_aware: bool = True
    target_aware: bool = True
    success_radius: float | None = None


@dataclass
class PointMazePolicyConfig:
    motion_dt: float = 0.02
    angular_velocity_decay: float = 0.26
    angular_velocity_mean: float = 0.0
    angular_velocity_std: float = 1.6336
    speed_mean: float = 0.25
    speed_std: float = 0.08
    speed_max: float = 0.35
    velocity_tracking_gain: float = 4.0
    action_smoothing: float = 0.3
    max_action_delta: float = 0.25
    touch_tangent_weight: float = 0.65
    touch_away_weight: float = 0.35
    touch_jitter_angle: float = 0.35 * 3.141592653589793
    stuck_threshold: float = 1e-4
    stuck_patience: int = 10


@dataclass
class PointMazeOutputConfig:
    output_dir: str = "data/datasets/pointmaze"
    dataset_name: str = "pointmaze_mujoco"
    episodes_per_shard: int = 1000
    num_workers: int = 1
    storage_format: str = "npz"


@dataclass
class PointMazeDatasetConfig:
    env: PointMazeEnvConfig = field(default_factory=PointMazeEnvConfig)
    policy: PointMazePolicyConfig = field(default_factory=PointMazePolicyConfig)
    output: PointMazeOutputConfig = field(default_factory=PointMazeOutputConfig)
    dataset_seed: int = 0
    num_episodes: int = 1000


def make_phase1_pointmaze_pi_env_config(*, max_episode_steps: int = 1000) -> PointMazeEnvConfig:
    return PointMazeEnvConfig(
        maze_map_name="OPEN",
        xml_file_path=default_phase1_pointmaze_xml_file_path(),
        continuing_task=False,
        reset_target=True,
        max_episode_steps=max_episode_steps,
        sensor_aware=True,
        achieved_goal_aware=True,
        start_pos_aware=True,
        target_aware=True,
        success_radius=0.4,
    )
