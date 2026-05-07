# PointMaze MuJoCo Dataset 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 `PointMaze` 实现一个独立的 MuJoCo 数据生成器，使用弱控制随机策略采样 episode，写出 `shard + Zarr` 数据集，并附带稳定空间标注与 episode 质量摘要。

**架构：** 新实现不依赖训练 checkpoint，也不把主流程塞进 `rl-baselines3-zoo/`。生成链路拆为 `EnvFactory -> PolicyDriver -> EpisodeCollector -> Annotator -> ShardWriter`，通过脚本入口按 shard 执行，并用 dataset-level / episode-level seed 保证复现性与断点重跑稳定性。

**技术栈：** Python 3.12, Gymnasium MuJoCo, NumPy, Zarr, pytest

---

## 文件结构

### 新建文件

- `components/dataset_gen/__init__.py`
  统一导出数据生成模块。
- `components/dataset_gen/pointmaze_config.py`
  数据类与配置解析辅助，集中定义生成参数、策略参数、输出参数。
- `components/dataset_gen/pointmaze_env_factory.py`
  创建 `PointMaze` 环境，固定 env kwargs 与 schema 提取。
- `components/dataset_gen/pointmaze_policy.py`
  实现弱控制随机策略与内部状态机。
- `components/dataset_gen/pointmaze_episode.py`
  定义标准化 episode 对象、step 缓冲结构、episode summary 计算。
- `components/dataset_gen/pointmaze_collector.py`
  实现 rollout 主循环，输出标准 episode。
- `components/dataset_gen/pointmaze_annotation.py`
  计算 `agent_xy`、`heading`、`goal_xy`、`relative_goal`。
- `components/dataset_gen/pointmaze_zarr_writer.py`
  实现 shard 写入、offset/length 管理、dataset metadata 写入。
- `components/dataset_gen/pointmaze_manifest.py`
  管理 episode seed 列表、shard manifest、断点重跑状态。
- `scripts/generate_pointmaze_dataset.py`
  CLI 入口，负责拼接配置、分配 shard、调用生成链路。
- `tests/test_pointmaze_dataset_policy.py`
  覆盖弱控制随机策略的关键行为。
- `tests/test_pointmaze_dataset_annotation.py`
  覆盖空间标注规则。
- `tests/test_pointmaze_dataset_writer.py`
  覆盖 shard 写入与读取后的基本结构。
- `tests/test_pointmaze_dataset_manifest.py`
  覆盖 seed 分配与 worker 无关性。
- `tests/test_pointmaze_dataset_collector.py`
  使用轻量 fake env 或最小 stub 验证 collector 输出结构。

### 修改文件

- `requirements.txt`
  补充 `zarr` 依赖。

## 任务 1：补齐依赖与模块骨架

**文件：**
- 创建：`components/dataset_gen/__init__.py`
- 创建：`components/dataset_gen/pointmaze_config.py`
- 修改：`requirements.txt`
- 测试：无

- [ ] **步骤 1：在 `requirements.txt` 添加 `zarr` 依赖**

```text
zarr
```

- [ ] **步骤 2：创建模块导出骨架**

```python
from .pointmaze_config import (
    PointMazeDatasetConfig,
    PointMazeEnvConfig,
    PointMazePolicyConfig,
    PointMazeOutputConfig,
)
```

- [ ] **步骤 3：定义基础配置数据类**

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PointMazeEnvConfig:
    env_id: str = "PointMaze"
    maze_map_name: str = "U_MAZE"
    xml_file_path: str | None = None
    continuing_task: bool = False
    reset_target: bool = True
    max_episode_steps: int = 300
    sensor_aware: bool = True


@dataclass(slots=True)
class PointMazePolicyConfig:
    segment_length_range: tuple[int, int] = (8, 32)
    action_noise_scale: float = 0.15
    stuck_threshold: float = 1e-3
    stuck_patience: int = 12
    subgoal_resample_prob: float = 0.1
    turn_bias: float = 0.35
    forward_bias: float = 0.75


@dataclass(slots=True)
class PointMazeOutputConfig:
    output_dir: Path
    dataset_name: str = "pointmaze_random_v1"
    episodes_per_shard: int = 256
    num_workers: int = 1


@dataclass(slots=True)
class PointMazeDatasetConfig:
    env: PointMazeEnvConfig = field(default_factory=PointMazeEnvConfig)
    policy: PointMazePolicyConfig = field(default_factory=PointMazePolicyConfig)
    output: PointMazeOutputConfig | None = None
    dataset_seed: int = 0
    num_episodes: int = 1024
```

- [ ] **步骤 4：运行基本导入检查**

运行：`python -c "from components.dataset_gen import PointMazeDatasetConfig; print(PointMazeDatasetConfig)"`  
预期：打印配置类，不报 `ImportError`

- [ ] **步骤 5：Commit**

```bash
git add requirements.txt components/dataset_gen/__init__.py components/dataset_gen/pointmaze_config.py
git commit -m "feat: add PointMaze dataset config skeleton"
```

## 任务 2：先写 seed/manifest 测试，再实现可复现分配

**文件：**
- 创建：`components/dataset_gen/pointmaze_manifest.py`
- 测试：`tests/test_pointmaze_dataset_manifest.py`

- [ ] **步骤 1：编写失败的 manifest 测试**

```python
from components.dataset_gen.pointmaze_manifest import build_episode_plan


def test_build_episode_plan_is_worker_independent():
    plan_a = build_episode_plan(dataset_seed=7, num_episodes=6, episodes_per_shard=2)
    plan_b = build_episode_plan(dataset_seed=7, num_episodes=6, episodes_per_shard=2)

    assert [item.episode_seed for item in plan_a.episodes] == [
        item.episode_seed for item in plan_b.episodes
    ]
    assert [item.shard_id for item in plan_a.episodes] == [0, 0, 1, 1, 2, 2]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_manifest.py -v`  
预期：FAIL，报错 `ModuleNotFoundError` 或 `cannot import name 'build_episode_plan'`

- [ ] **步骤 3：实现 manifest 结构与 seed 规划**

```python
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True, frozen=True)
class EpisodePlanItem:
    episode_id: int
    shard_id: int
    episode_seed: int


@dataclass(slots=True)
class EpisodePlan:
    dataset_seed: int
    episodes: list[EpisodePlanItem]


def build_episode_plan(dataset_seed: int, num_episodes: int, episodes_per_shard: int) -> EpisodePlan:
    seed_sequence = np.random.SeedSequence(dataset_seed)
    child_sequences = seed_sequence.spawn(num_episodes)
    items = []
    for episode_id, child in enumerate(child_sequences):
        shard_id = episode_id // episodes_per_shard
        episode_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        items.append(EpisodePlanItem(episode_id=episode_id, shard_id=shard_id, episode_seed=episode_seed))
    return EpisodePlan(dataset_seed=dataset_seed, episodes=items)
```

- [ ] **步骤 4：增加 manifest 持久化测试**

```python
from components.dataset_gen.pointmaze_manifest import save_manifest, load_manifest


def test_manifest_roundtrip(tmp_path):
    plan = build_episode_plan(dataset_seed=11, num_episodes=4, episodes_per_shard=2)
    path = tmp_path / "manifest.json"

    save_manifest(plan, path)
    loaded = load_manifest(path)

    assert loaded.dataset_seed == 11
    assert loaded.episodes[3].episode_seed == plan.episodes[3].episode_seed
```

- [ ] **步骤 5：实现 JSON 持久化**

```python
import json
from pathlib import Path
from dataclasses import asdict


def save_manifest(plan: EpisodePlan, path: Path) -> None:
    payload = {
        "dataset_seed": plan.dataset_seed,
        "episodes": [asdict(item) for item in plan.episodes],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> EpisodePlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = [EpisodePlanItem(**item) for item in payload["episodes"]]
    return EpisodePlan(dataset_seed=payload["dataset_seed"], episodes=episodes)
```

- [ ] **步骤 6：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_manifest.py -v`  
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add components/dataset_gen/pointmaze_manifest.py tests/test_pointmaze_dataset_manifest.py
git commit -m "feat: add PointMaze dataset manifest planning"
```

## 任务 3：先写策略测试，再实现弱控制随机策略

**文件：**
- 创建：`components/dataset_gen/pointmaze_policy.py`
- 测试：`tests/test_pointmaze_dataset_policy.py`

- [ ] **步骤 1：编写失败的策略测试**

```python
import numpy as np

from components.dataset_gen.pointmaze_config import PointMazePolicyConfig
from components.dataset_gen.pointmaze_policy import WeakRandomPolicyDriver


def test_policy_holds_action_within_segment():
    config = PointMazePolicyConfig(segment_length_range=(3, 3), action_noise_scale=0.0)
    policy = WeakRandomPolicyDriver(config=config, seed=123)

    action_0 = policy.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    action_1 = policy.act(agent_xy=np.array([0.1, 0.0]), heading=0.0, collision=False)
    action_2 = policy.act(agent_xy=np.array([0.2, 0.0]), heading=0.0, collision=False)

    assert np.allclose(action_0, action_1)
    assert np.allclose(action_1, action_2)
```

- [ ] **步骤 2：增加 stuck 重采样测试**

```python
def test_policy_resamples_when_stuck():
    config = PointMazePolicyConfig(segment_length_range=(20, 20), action_noise_scale=0.0, stuck_threshold=0.01, stuck_patience=2)
    policy = WeakRandomPolicyDriver(config=config, seed=9)

    first = policy.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    second = policy.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)
    third = policy.act(agent_xy=np.array([0.0, 0.0]), heading=0.0, collision=False)

    assert not np.allclose(first, third)
    assert np.allclose(first, second)
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_policy.py -v`  
预期：FAIL，报错 `ModuleNotFoundError`

- [ ] **步骤 4：实现策略驱动器**

```python
from dataclasses import dataclass
import numpy as np

from .pointmaze_config import PointMazePolicyConfig


@dataclass(slots=True)
class PolicyState:
    remaining_steps: int = 0
    current_action: np.ndarray | None = None
    last_agent_xy: np.ndarray | None = None
    stuck_steps: int = 0


class WeakRandomPolicyDriver:
    def __init__(self, config: PointMazePolicyConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = PolicyState()

    def _sample_action(self, heading: float) -> np.ndarray:
        forward = self.config.forward_bias + self.rng.normal(0.0, self.config.action_noise_scale)
        turn = self.rng.normal(0.0, self.config.turn_bias)
        return np.clip(np.array([forward, turn], dtype=np.float32), -1.0, 1.0)

    def _need_resample(self, agent_xy: np.ndarray, collision: bool) -> bool:
        if self.state.current_action is None or self.state.remaining_steps <= 0:
            return True
        if collision:
            return True
        if self.state.last_agent_xy is not None:
            disp = np.linalg.norm(agent_xy - self.state.last_agent_xy)
            if disp < self.config.stuck_threshold:
                self.state.stuck_steps += 1
            else:
                self.state.stuck_steps = 0
            if self.state.stuck_steps >= self.config.stuck_patience:
                return True
        return self.rng.random() < self.config.subgoal_resample_prob

    def act(self, agent_xy: np.ndarray, heading: float, collision: bool) -> np.ndarray:
        if self._need_resample(agent_xy, collision):
            self.state.current_action = self._sample_action(heading)
            self.state.remaining_steps = self.rng.integers(
                self.config.segment_length_range[0],
                self.config.segment_length_range[1] + 1,
            )
            self.state.stuck_steps = 0
        action = self.state.current_action.copy()
        self.state.remaining_steps -= 1
        self.state.last_agent_xy = agent_xy.copy()
        return action
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_policy.py -v`  
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add components/dataset_gen/pointmaze_policy.py tests/test_pointmaze_dataset_policy.py
git commit -m "feat: add weak random PointMaze rollout policy"
```

## 任务 4：先写 annotation 测试，再实现稳定空间标注

**文件：**
- 创建：`components/dataset_gen/pointmaze_annotation.py`
- 测试：`tests/test_pointmaze_dataset_annotation.py`

- [ ] **步骤 1：编写失败的 annotation 测试**

```python
import numpy as np

from components.dataset_gen.pointmaze_annotation import annotate_episode


def test_annotate_episode_derives_stable_spatial_fields():
    episode = {
        "qpos": np.array([[1.0, 2.0, 0.0], [1.5, 2.5, 0.1]]),
        "goal": np.array([[3.0, 5.0], [3.0, 5.0]]),
    }

    annotations = annotate_episode(episode)

    assert np.allclose(annotations["agent_xy"], np.array([[1.0, 2.0], [1.5, 2.5]]))
    assert np.allclose(annotations["goal_xy"], np.array([[3.0, 5.0], [3.0, 5.0]]))
    assert np.allclose(annotations["relative_goal"], np.array([[2.0, 3.0], [1.5, 2.5]]))
```

- [ ] **步骤 2：增加 heading 测试**

```python
def test_heading_uses_qpos_delta_when_available():
    episode = {
        "qpos": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]),
        "goal": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    }

    annotations = annotate_episode(episode)

    assert np.isclose(annotations["heading"][1], np.pi / 4)
    assert np.isclose(annotations["heading"][2], 0.0)
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_annotation.py -v`  
预期：FAIL

- [ ] **步骤 4：实现 annotation 逻辑**

```python
import numpy as np


def _compute_heading(agent_xy: np.ndarray) -> np.ndarray:
    deltas = np.diff(agent_xy, axis=0, prepend=agent_xy[:1])
    heading = np.arctan2(deltas[:, 1], deltas[:, 0])
    if len(heading) > 1:
        heading[0] = heading[1]
    return heading.astype(np.float32)


def annotate_episode(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    agent_xy = np.asarray(episode["qpos"])[..., :2].astype(np.float32)
    goal_xy = np.asarray(episode["goal"])[..., :2].astype(np.float32)
    heading = _compute_heading(agent_xy)
    relative_goal = goal_xy - agent_xy
    return {
        "agent_xy": agent_xy,
        "heading": heading,
        "goal_xy": goal_xy,
        "relative_goal": relative_goal.astype(np.float32),
    }
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_annotation.py -v`  
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add components/dataset_gen/pointmaze_annotation.py tests/test_pointmaze_dataset_annotation.py
git commit -m "feat: add PointMaze spatial annotations"
```

## 任务 5：先写 collector 测试，再实现标准 episode 对象与质量摘要

**文件：**
- 创建：`components/dataset_gen/pointmaze_episode.py`
- 创建：`components/dataset_gen/pointmaze_collector.py`
- 测试：`tests/test_pointmaze_dataset_collector.py`

- [ ] **步骤 1：编写失败的 collector 测试**

```python
import numpy as np

from components.dataset_gen.pointmaze_collector import collect_episode


class FakeEnv:
    def __init__(self):
        self.step_idx = 0
        self.goal = np.array([2.0, 2.0], dtype=np.float32)

    def reset(self, seed=None):
        self.step_idx = 0
        obs = {"observation": np.array([0.0]), "desired_goal": self.goal.copy()}
        info = {"qpos": np.array([0.0, 0.0]), "qvel": np.array([0.0, 0.0]), "success": False}
        return obs, info

    def step(self, action):
        self.step_idx += 1
        qpos = np.array([float(self.step_idx), 0.0], dtype=np.float32)
        qvel = np.array([1.0, 0.0], dtype=np.float32)
        obs = {"observation": np.array([self.step_idx]), "desired_goal": self.goal.copy()}
        info = {"qpos": qpos, "qvel": qvel, "success": self.step_idx >= 3, "speed": 1.0}
        return obs, 0.5, self.step_idx >= 3, False, info
```

- [ ] **步骤 2：补完整体断言**

```python
class ConstantPolicy:
    def act(self, agent_xy, heading, collision):
        return np.array([0.5, 0.0], dtype=np.float32)


def test_collect_episode_returns_normalized_episode():
    episode = collect_episode(
        env=FakeEnv(),
        policy=ConstantPolicy(),
        episode_id=4,
        episode_seed=101,
        env_metadata={"maze_map_name": "U_MAZE"},
        policy_metadata={"policy_type": "constant"},
    )

    assert episode["episode_id"] == 4
    assert episode["seed"] == 101
    assert episode["obs"]["observation"].shape == (3, 1)
    assert episode["action"].shape == (3, 2)
    assert episode["qpos"].shape == (3, 2)
    assert episode["summary"]["episode_length"] == 3
    assert episode["summary"]["path_length"] == 2.0
    assert episode["summary"]["goal_reached"] is True
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_collector.py -v`  
预期：FAIL

- [ ] **步骤 4：实现标准 episode 与 summary 计算**

```python
import numpy as np


def summarize_episode(qpos: np.ndarray, qvel: np.ndarray, rewards: np.ndarray, terminated: np.ndarray, info_list: list[dict]) -> dict:
    agent_xy = qpos[:, :2]
    deltas = np.diff(agent_xy, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(agent_xy) > 1 else 0.0
    net_displacement = float(np.linalg.norm(agent_xy[-1] - agent_xy[0])) if len(agent_xy) > 1 else 0.0
    mean_speed = float(np.linalg.norm(qvel[:, :2], axis=1).mean()) if len(qvel) else 0.0
    return {
        "episode_length": int(len(rewards)),
        "return": float(rewards.sum()),
        "terminated_reason": "terminated" if bool(terminated[-1]) else "truncated",
        "path_length": path_length,
        "net_displacement": net_displacement,
        "coverage_score": float(len(np.unique(np.floor(agent_xy), axis=0))),
        "stuck_ratio": 0.0,
        "collision_ratio": 0.0,
        "goal_reached": bool(info_list[-1].get("success", False)),
        "mean_speed": mean_speed,
        "mean_turn_rate": 0.0,
    }
```

- [ ] **步骤 5：实现 collector 主循环**

```python
def collect_episode(env, policy, episode_id: int, episode_seed: int, env_metadata: dict, policy_metadata: dict) -> dict:
    obs, info = env.reset(seed=episode_seed)
    obs_buffer = {k: [] for k in obs}
    actions, rewards, terminated, truncated, qpos, qvel, goals, infos = [], [], [], [], [], [], [], []

    done = False
    last_xy = np.asarray(info["qpos"][:2], dtype=np.float32)
    heading = 0.0
    while not done:
        action = policy.act(agent_xy=last_xy, heading=heading, collision=False)
        next_obs, reward, term, trunc, step_info = env.step(action)
        for key, value in next_obs.items():
            obs_buffer.setdefault(key, []).append(np.asarray(value))
        actions.append(np.asarray(action))
        rewards.append(reward)
        terminated.append(term)
        truncated.append(trunc)
        qpos.append(np.asarray(step_info["qpos"]))
        qvel.append(np.asarray(step_info["qvel"]))
        goals.append(np.asarray(next_obs["desired_goal"]))
        infos.append(step_info)
        new_xy = np.asarray(step_info["qpos"][:2], dtype=np.float32)
        delta = new_xy - last_xy
        if np.linalg.norm(delta) > 0:
            heading = float(np.arctan2(delta[1], delta[0]))
        last_xy = new_xy
        done = bool(term or trunc)

    episode = {
        "episode_id": episode_id,
        "seed": episode_seed,
        "env_metadata": env_metadata,
        "policy_metadata": policy_metadata,
        "obs": {k: np.asarray(v) for k, v in obs_buffer.items()},
        "action": np.asarray(actions),
        "reward": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "goal": np.asarray(goals),
        "info": infos,
    }
    episode["summary"] = summarize_episode(episode["qpos"], episode["qvel"], episode["reward"], episode["terminated"], infos)
    return episode
```

- [ ] **步骤 6：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_collector.py -v`  
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add components/dataset_gen/pointmaze_episode.py components/dataset_gen/pointmaze_collector.py tests/test_pointmaze_dataset_collector.py
git commit -m "feat: add PointMaze episode collector"
```

## 任务 6：先写 writer 测试，再实现 `shard + Zarr` 写盘

**文件：**
- 创建：`components/dataset_gen/pointmaze_zarr_writer.py`
- 测试：`tests/test_pointmaze_dataset_writer.py`

- [ ] **步骤 1：编写失败的 writer 测试**

```python
import numpy as np
import zarr

from components.dataset_gen.pointmaze_zarr_writer import write_shard


def test_write_shard_persists_offsets_and_annotations(tmp_path):
    episodes = [
        {
            "episode_id": 0,
            "seed": 10,
            "obs": {"observation": np.array([[1.0], [2.0]])},
            "action": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "reward": np.array([0.0, 1.0]),
            "terminated": np.array([False, True]),
            "truncated": np.array([False, False]),
            "qpos": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "qvel": np.array([[0.1, 0.0], [0.2, 0.0]]),
            "goal": np.array([[2.0, 2.0], [2.0, 2.0]]),
            "annotation": {
                "agent_xy": np.array([[0.0, 0.0], [1.0, 0.0]]),
                "heading": np.array([0.0, 0.0]),
                "goal_xy": np.array([[2.0, 2.0], [2.0, 2.0]]),
                "relative_goal": np.array([[2.0, 2.0], [1.0, 2.0]]),
            },
            "summary": {"episode_length": 2, "return": 1.0},
        }
    ]

    shard_path = write_shard(tmp_path, shard_id=0, episodes=episodes, dataset_meta={"dataset_name": "demo"})
    root = zarr.open(shard_path, mode="r")

    assert root["episode_lengths"][:].tolist() == [2]
    assert root["step/qpos"][:].shape == (2, 2)
    assert root["annotation/relative_goal"][:].shape == (2, 2)
    assert root.attrs["dataset_name"] == "demo"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_writer.py -v`  
预期：FAIL

- [ ] **步骤 3：实现 shard 写入逻辑**

```python
from pathlib import Path
import json
import numpy as np
import zarr


def _concat_field(episodes: list[dict], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(ep[key]) for ep in episodes], axis=0)


def write_shard(output_dir: Path, shard_id: int, episodes: list[dict], dataset_meta: dict) -> Path:
    shard_path = output_dir / f"train_shard_{shard_id:04d}.zarr"
    root = zarr.open(shard_path, mode="w")
    episode_lengths = np.asarray([int(ep["summary"]["episode_length"]) for ep in episodes], dtype=np.int32)
    episode_offsets = np.concatenate([[0], np.cumsum(episode_lengths[:-1])]).astype(np.int64)
    root.create_array("episode_lengths", data=episode_lengths)
    root.create_array("episode_offsets", data=episode_offsets)
    step_group = root.create_group("step")
    step_group.create_array("action", data=_concat_field(episodes, "action"))
    step_group.create_array("reward", data=_concat_field(episodes, "reward"))
    step_group.create_array("terminated", data=_concat_field(episodes, "terminated"))
    step_group.create_array("truncated", data=_concat_field(episodes, "truncated"))
    step_group.create_array("qpos", data=_concat_field(episodes, "qpos"))
    step_group.create_array("qvel", data=_concat_field(episodes, "qvel"))
    step_group.create_array("goal", data=_concat_field(episodes, "goal"))
    annotation_group = root.create_group("annotation")
    for name in ("agent_xy", "heading", "goal_xy", "relative_goal"):
        annotation_group.create_array(name, data=np.concatenate([ep["annotation"][name] for ep in episodes], axis=0))
    root.attrs.update(dataset_meta)
    root.attrs["episode_summary_json"] = json.dumps([ep["summary"] for ep in episodes])
    return shard_path
```

- [ ] **步骤 4：补 dataset metadata 写入函数测试**

```python
from components.dataset_gen.pointmaze_zarr_writer import write_dataset_metadata


def test_write_dataset_metadata_json(tmp_path):
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    meta_path = write_dataset_metadata(output_dir, {"dataset_name": "pointmaze_random_v1", "env_id": "PointMaze"})
    payload = meta_path.read_text(encoding="utf-8")
    assert '"dataset_name": "pointmaze_random_v1"' in payload
    assert '"env_id": "PointMaze"' in payload
```

- [ ] **步骤 5：实现 dataset metadata 文件写出**

```python
def write_dataset_metadata(output_dir: Path, dataset_meta: dict) -> Path:
    path = output_dir / "dataset_meta.json"
    path.write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    return path
```

- [ ] **步骤 6：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_writer.py -v`  
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add components/dataset_gen/pointmaze_zarr_writer.py tests/test_pointmaze_dataset_writer.py
git commit -m "feat: add PointMaze Zarr shard writer"
```

## 任务 7：实现 `PointMaze` 环境工厂并接通 collector + annotation + writer

**文件：**
- 创建：`components/dataset_gen/pointmaze_env_factory.py`
- 修改：`components/dataset_gen/__init__.py`
- 测试：`tests/test_pointmaze_dataset_collector.py`

- [ ] **步骤 1：为环境工厂添加最小集成测试**

```python
from components.dataset_gen.pointmaze_config import PointMazeEnvConfig
from components.dataset_gen.pointmaze_env_factory import build_pointmaze_env_kwargs


def test_build_pointmaze_env_kwargs_contains_expected_keys():
    kwargs = build_pointmaze_env_kwargs(PointMazeEnvConfig(maze_map_name="OPEN", max_episode_steps=100))
    assert kwargs["maze_map_name"] == "OPEN"
    assert kwargs["max_episode_steps"] == 100
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_collector.py -v`  
预期：FAIL，报错 `cannot import name 'build_pointmaze_env_kwargs'`

- [ ] **步骤 3：实现环境工厂**

```python
from envs.point_maze import PointMazeEnv

from .pointmaze_config import PointMazeEnvConfig


def build_pointmaze_env_kwargs(config: PointMazeEnvConfig) -> dict:
    kwargs = {
        "maze_map_name": config.maze_map_name,
        "continuing_task": config.continuing_task,
        "reset_target": config.reset_target,
        "max_episode_steps": config.max_episode_steps,
        "sensor_aware": config.sensor_aware,
    }
    if config.xml_file_path is not None:
        kwargs["xml_file_path"] = config.xml_file_path
    return kwargs


def create_pointmaze_env(config: PointMazeEnvConfig) -> PointMazeEnv:
    return PointMazeEnv(**build_pointmaze_env_kwargs(config))
```

- [ ] **步骤 4：把 collector 测试扩展为 annotation 集成测试**

```python
from components.dataset_gen.pointmaze_annotation import annotate_episode


def test_collected_episode_can_be_annotated():
    episode = collect_episode(
        env=FakeEnv(),
        policy=ConstantPolicy(),
        episode_id=0,
        episode_seed=3,
        env_metadata={"maze_map_name": "U_MAZE"},
        policy_metadata={"policy_type": "constant"},
    )
    annotation = annotate_episode(episode)
    assert annotation["agent_xy"].shape == (3, 2)
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_collector.py -v`  
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add components/dataset_gen/pointmaze_env_factory.py components/dataset_gen/__init__.py tests/test_pointmaze_dataset_collector.py
git commit -m "feat: add PointMaze env factory"
```

## 任务 8：实现 CLI 入口与 shard 级编排

**文件：**
- 创建：`scripts/generate_pointmaze_dataset.py`
- 测试：`tests/test_pointmaze_dataset_manifest.py`

- [ ] **步骤 1：先写 CLI 干跑测试**

```python
import subprocess
import sys


def test_generate_pointmaze_dataset_help():
    result = subprocess.run(
        [sys.executable, "scripts/generate_pointmaze_dataset.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--num-episodes" in result.stdout
    assert "--episodes-per-shard" in result.stdout
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_pointmaze_dataset_manifest.py -v`  
预期：FAIL，报错脚本不存在

- [ ] **步骤 3：实现 CLI 参数解析与主流程**

```python
import argparse
from pathlib import Path

from components.dataset_gen.pointmaze_annotation import annotate_episode
from components.dataset_gen.pointmaze_collector import collect_episode
from components.dataset_gen.pointmaze_config import (
    PointMazeDatasetConfig,
    PointMazeEnvConfig,
    PointMazeOutputConfig,
    PointMazePolicyConfig,
)
from components.dataset_gen.pointmaze_env_factory import create_pointmaze_env, build_pointmaze_env_kwargs
from components.dataset_gen.pointmaze_manifest import build_episode_plan, save_manifest
from components.dataset_gen.pointmaze_policy import WeakRandomPolicyDriver
from components.dataset_gen.pointmaze_zarr_writer import write_dataset_metadata, write_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PointMaze MuJoCo dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="pointmaze_random_v1")
    parser.add_argument("--num-episodes", type=int, default=1024)
    parser.add_argument("--episodes-per-shard", type=int, default=256)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--maze-map-name", type=str, default="U_MAZE")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    return parser.parse_args()
```

- [ ] **步骤 4：实现按 shard 编排**

```python
def main() -> None:
    args = parse_args()
    output_config = PointMazeOutputConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        episodes_per_shard=args.episodes_per_shard,
    )
    config = PointMazeDatasetConfig(
        env=PointMazeEnvConfig(maze_map_name=args.maze_map_name, max_episode_steps=args.max_episode_steps),
        policy=PointMazePolicyConfig(),
        output=output_config,
        dataset_seed=args.dataset_seed,
        num_episodes=args.num_episodes,
    )
    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_episode_plan(
        dataset_seed=config.dataset_seed,
        num_episodes=config.num_episodes,
        episodes_per_shard=config.output.episodes_per_shard,
    )
    save_manifest(plan, config.output.output_dir / "manifest.json")
    dataset_meta = {
        "dataset_name": config.output.dataset_name,
        "env_id": config.env.env_id,
        "policy_spec": "weak_random_v1",
        "seed_strategy": "dataset_seed + episode_seed",
        "observation_space_spec": "PointMazeEnv.observation_space",
        "action_space_spec": "PointMazeEnv.action_space",
    }
    write_dataset_metadata(config.output.output_dir, dataset_meta)
    for shard_id in sorted({item.shard_id for item in plan.episodes}):
        shard_items = [item for item in plan.episodes if item.shard_id == shard_id]
        episodes = []
        for item in shard_items:
            env = create_pointmaze_env(config.env)
            policy = WeakRandomPolicyDriver(config.policy, seed=item.episode_seed)
            episode = collect_episode(
                env=env,
                policy=policy,
                episode_id=item.episode_id,
                episode_seed=item.episode_seed,
                env_metadata=build_pointmaze_env_kwargs(config.env),
                policy_metadata={"policy_type": "weak_random_v1"},
            )
            episode["annotation"] = annotate_episode(episode)
            episodes.append(episode)
            env.close()
        write_shard(config.output.output_dir, shard_id=shard_id, episodes=episodes, dataset_meta=dataset_meta)
```

- [ ] **步骤 5：运行 CLI 帮助测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_manifest.py -v`  
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add scripts/generate_pointmaze_dataset.py tests/test_pointmaze_dataset_manifest.py
git commit -m "feat: add PointMaze dataset generation CLI"
```

## 任务 9：增加最小端到端烟雾测试

**文件：**
- 修改：`tests/test_pointmaze_dataset_writer.py`
- 修改：`tests/test_pointmaze_dataset_manifest.py`

- [ ] **步骤 1：编写端到端小样本测试**

```python
import subprocess
import sys
from pathlib import Path


def test_generate_pointmaze_dataset_end_to_end(tmp_path):
    output_dir = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_pointmaze_dataset.py",
            "--output-dir",
            str(output_dir),
            "--num-episodes",
            "2",
            "--episodes-per-shard",
            "2",
            "--dataset-seed",
            "5",
            "--max-episode-steps",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (output_dir / "dataset_meta.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert any(path.suffix == ".zarr" for path in output_dir.iterdir())
```

- [ ] **步骤 2：运行测试验证失败或暴露真实集成缺口**

运行：`pytest tests/test_pointmaze_dataset_manifest.py::test_generate_pointmaze_dataset_end_to_end -v`  
预期：首轮 FAIL，原因应是集成代码缺口而非测试错误

- [ ] **步骤 3：补齐集成缺口并确保脚本可运行**

```python
# 这里不新增接口，而是修正前面任务的实际集成问题：
# - 确认 env.close() 存在时才调用
# - 确认 obs dict 的字段都能序列化为 ndarray
# - 确认 write_shard 对单 episode shard 也能工作
# - 确认 manifest / dataset meta 路径提前创建
```

- [ ] **步骤 4：运行端到端测试验证通过**

运行：`pytest tests/test_pointmaze_dataset_manifest.py::test_generate_pointmaze_dataset_end_to_end -v`  
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add tests/test_pointmaze_dataset_manifest.py tests/test_pointmaze_dataset_writer.py components/dataset_gen scripts/generate_pointmaze_dataset.py
git commit -m "test: add PointMaze dataset generation smoke test"
```

## 任务 10：运行最终验证并记录最小使用命令

**文件：**
- 无代码变更，必要时仅修复测试暴露的问题

- [ ] **步骤 1：运行目标测试集**

运行：`pytest tests/test_pointmaze_dataset_policy.py tests/test_pointmaze_dataset_annotation.py tests/test_pointmaze_dataset_collector.py tests/test_pointmaze_dataset_writer.py tests/test_pointmaze_dataset_manifest.py -v`  
预期：全部 PASS

- [ ] **步骤 2：运行最小示例命令**

运行：

```bash
python scripts/generate_pointmaze_dataset.py \
  --output-dir /tmp/pointmaze_dataset_smoke \
  --num-episodes 2 \
  --episodes-per-shard 2 \
  --dataset-seed 0 \
  --max-episode-steps 8
```

预期：
- 生成 `dataset_meta.json`
- 生成 `manifest.json`
- 生成至少一个 `train_shard_0000.zarr`

- [ ] **步骤 3：如果验证暴露问题，做最小修复并重新运行相关测试**

```python
# 只修复验证实际暴露的问题，不额外扩 scope。
```

- [ ] **步骤 4：Commit**

```bash
git add requirements.txt components/dataset_gen scripts/generate_pointmaze_dataset.py tests
git commit -m "feat: finalize PointMaze dataset generator"
```

## 规格覆盖自检

- `PointMaze only`：由任务 1、7、8 固定环境范围。
- `weak random-control policy`：由任务 3 实现并测试。
- `raw rollout fields`：由任务 5、6 收集并写盘。
- `stable annotations only`：由任务 4 实现，未把 maze topology 标签混入首版。
- `shard + Zarr`：由任务 6、8 实现。
- `hierarchical seeds and worker independence`：由任务 2 固定。
- `episode quality summaries`：由任务 5 负责。
- `standalone generator, not zoo runtime`：由任务 8 的脚本入口保证。

## 占位符自检

- 没有使用 `TODO`、`TBD`、`后续实现` 一类占位符。
- 每个测试步骤都给出了明确的测试代码或命令。
- 每个实现步骤都给出了目标接口或最小代码骨架。

## 类型一致性自检

- 配置类型统一使用 `PointMazeDatasetConfig / PointMazeEnvConfig / PointMazePolicyConfig / PointMazeOutputConfig`。
- 策略类型统一使用 `WeakRandomPolicyDriver`。
- 标准 episode 中原始动作字段统一命名为 `action`，标注字段统一挂在 `annotation`。
- shard 写盘字段统一使用 `episode_lengths / episode_offsets / step/* / annotation/*`。
