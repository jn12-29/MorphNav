from dataclasses import dataclass, field


@dataclass
class PointMazeEnvConfig:
    env_id: str = "PointMaze"
    maze_map_name: str = "OPEN"
    xml_file_path: str = ""
    continuing_task: bool = True
    reset_target: bool = False
    # Collection-side safety cap; applied by dataset collector, not env constructor kwargs.
    max_episode_steps: int = 1000
    sensor_aware: bool = False


@dataclass
class PointMazePolicyConfig:
    segment_length_range: tuple[int, int] = (16, 64)
    action_noise_scale: float = 0.1
    stuck_threshold: float = 0.05
    stuck_patience: int = 10
    subgoal_resample_prob: float = 0.2
    turn_bias: float = 0.0
    forward_bias: float = 1.0


@dataclass
class PointMazeOutputConfig:
    output_dir: str = "recorded_data/pointmaze"
    dataset_name: str = "pointmaze_mujoco"
    episodes_per_shard: int = 100
    num_workers: int = 1


@dataclass
class PointMazeDatasetConfig:
    env: PointMazeEnvConfig = field(default_factory=PointMazeEnvConfig)
    policy: PointMazePolicyConfig = field(default_factory=PointMazePolicyConfig)
    output: PointMazeOutputConfig = field(default_factory=PointMazeOutputConfig)
    dataset_seed: int = 0
    num_episodes: int = 1000
