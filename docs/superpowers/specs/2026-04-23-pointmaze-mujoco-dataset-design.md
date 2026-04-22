# PointMaze MuJoCo Dataset Design

## Summary

This design defines a first-version MuJoCo dataset generator for `PointMaze`.

The dataset is intended for spatial representation learning, but the generator should first produce a high-fidelity reusable data asset instead of a task-specific training format. The generator only covers two layers: raw rollout collection and lightweight spatial annotation. Train/eval splitting, fixed-window export, and richer maze-topology labels remain downstream concerns.

## Goals

- Generate reproducible `PointMaze` MuJoCo episodes without depending on trained policies.
- Record rollout data with enough fidelity to support future reprocessing and relabeling.
- Add a small set of stable spatial annotations that are broadly useful for spatial representation learning.
- Store data in a shard-based format that scales beyond small prototype runs.
- Preserve dataset-level and episode-level metadata needed for auditing, filtering, and regeneration.

## Non-Goals

- Do not support `AntMaze` or other MuJoCo environments in the first version.
- Do not generate train/eval splits during data generation.
- Do not export fixed-length windows during data generation.
- Do not bake in maze-topology-heavy labels such as room ids, shortest-path progress, or wall distance maps.
- Do not rely on `rl-baselines3-zoo/` as the core generator runtime.

## Scope

The first version only targets `PointMaze` and only implements:

- environment creation
- weak-policy rollout generation
- raw episode recording
- lightweight spatial annotation
- shard-based dataset writing
- episode quality summaries
- reproducible seed management

Everything else is explicitly deferred.

## Design Overview

The generator should produce a reusable offline dataset asset with two layers:

1. `Rollout Layer`
   Collect full MuJoCo episodes using a weak random-control policy.
2. `Spatial Annotation Layer`
   Add a minimal set of stable spatial fields derived from recorded state.

The resulting dataset should be useful both as a direct source for spatial representation learning and as a base asset for future relabeling, slicing, filtering, and format conversion.

## Why This Shape

The repository already has MuJoCo environments and some recording utilities, but the desired dataset is not just an experiment log dump.

The chosen design deliberately favors:

- high-fidelity reusable recording over narrow task formatting
- lightweight, stable annotation over aggressive semantic preprocessing
- reproducibility over throughput shortcuts
- separation from the training stack so the generator can run without checkpoints

This keeps the first version focused while leaving room for downstream data products.

## Data Generation Strategy

### Environment

The first version should use `PointMaze` only.

Relevant environment settings such as `maze_map_name`, `xml_file_path`, `continuing_task`, and `max_episode_steps` must be explicitly recorded in dataset metadata and episode metadata when applicable.

### Rollout Policy

The rollout policy should be a weak random-control policy rather than per-step white-noise actions.

It should follow a simple “random intent plus short execution” pattern:

- sample an action intent or local movement tendency
- keep that intent for a short segment
- resample early when the agent is stuck, oscillating, or colliding repeatedly

This policy is not meant to solve the task. Its purpose is to improve spatial coverage and temporal coherence compared with naive independent random actions.

### Policy Controls

The policy configuration should expose a small set of tunable parameters:

- `segment_length_range`
- `action_noise_scale`
- `stuck_threshold`
- `stuck_patience`
- `subgoal_resample_prob`
- `turn_bias`
- `forward_bias`
- `max_episode_steps`

These parameters should be written to metadata so that dataset variants remain traceable.

## Module Boundaries

The generator should be split into four modules with clear responsibilities.

### 1. `EnvFactory`

Responsible for:

- constructing the `PointMaze` environment
- applying environment kwargs
- applying reset seeds
- exposing observation and action schema information

Not responsible for:

- rollout logic
- annotation
- persistence

### 2. `PolicyDriver`

Responsible for:

- producing the next action from current observation, limited history, and policy RNG state

Not responsible for:

- environment stepping
- writing data
- semantic annotation

This module isolates policy behavior so later dataset variants can swap in other policy distributions without changing the collector.

### 3. `EpisodeCollector`

Responsible for:

- resetting the environment
- stepping until termination or truncation
- collecting raw step data
- creating episode-level metadata and summaries

The collector should produce a normalized episode object with a stable field contract. It should not mix in high-level annotation logic.

### 4. `Annotator` and `ShardWriter`

`Annotator` is responsible for deriving stable spatial annotations from recorded episode content.

`ShardWriter` is responsible for persisting raw fields, annotations, and metadata to the dataset store.

These concerns should stay logically separate so annotation rules can evolve without changing rollout collection.

## Recorded Fields

### Raw Step Fields

Each episode should record at least:

- `obs`
- `action`
- `reward`
- `terminated`
- `truncated`
- `qpos`
- `qvel`
- `goal`
- selected serializable `info` fields

Optional but recommended when inexpensive to capture:

- `collision_flag` or equivalent contact summary
- `discount` or `mask`

The generator should prefer recording real engine state instead of reconstructing it later from partial logs.

### Episode Metadata

Each episode should include metadata such as:

- `episode_id`
- `env_id`
- `maze_map_name`
- `xml_file_path` or model identifier
- `max_episode_steps`
- `policy_type`
- `policy_version`
- `policy_params`
- `seed`
- `recorded_at`
- observation schema identifier
- state schema identifier

The exact schema can be compact, but it must be sufficient to audit or regenerate the dataset.

## Spatial Annotation Strategy

The first version should only persist stable annotations that are low-risk and broadly reusable:

- `agent_xy`
- `heading`
- `goal_xy`
- `relative_goal`

These annotations are intentionally conservative. They support spatial representation work without forcing the first generator version to commit to a maze-topology interpretation that may later change.

The following remain downstream derived annotations and should not be hard-coded into the first generator:

- `maze_cell`
- `distance_to_wall`
- `shortest_path_progress`
- `room_id`
- `junction_id`
- other topology-specific semantic labels

## Storage Design

The generator should use shard-based storage rather than one file per episode.

### Why Shards

Shards avoid the operational problems of many tiny files:

- poor file-system scalability
- slow dataset enumeration
- harder restart and deduplication logic
- higher cost for later aggregation and slicing

### Shard Structure

Each shard should contain many episodes, for example hundreds to low thousands depending on length.

Because episodes are variable length, shard content should include:

- concatenated step arrays
- `episode_lengths` or `episode_offsets`
- episode-level metadata table
- shard-level metadata

### Storage Format

The preferred long-term format is `Zarr`.

Reasoning:

- chunked array storage fits large rollout datasets well
- partial reads are better for future processing
- it scales better than compressed `.npz` once dataset size increases

`NPZ` may still be acceptable for a small prototype, but the design target for the first real implementation should be `shard + Zarr`.

### Dataset Metadata

At dataset root, there should be a dataset metadata file recording at least:

- `dataset_name`
- `dataset_version`
- `env_id`
- `field_schema`
- `annotation_schema`
- `policy_spec`
- `seed_strategy`
- `action_space_spec`
- `observation_space_spec`
- free-form notes

## Reproducibility

The generator must use hierarchical seed management.

### Seed Layers

1. `dataset_seed`
   Determines the full generation run.
2. `episode_seed`
   One independent seed per episode.
3. `worker execution`
   Workers consume preassigned episode seeds rather than generating their own random sequence.

### Required Properties

This design should guarantee:

- changing `num_workers` does not change episode content
- failed shards can be retried without changing successful episodes
- policy variants can be compared using the same episode seed list

This is a direct carryover from the reference logic where sample-level seeds are more important than worker-local RNG order.

## Quality Summaries

The first version should not aggressively discard data during collection, but it should compute lightweight quality summaries for each episode.

Recommended episode summaries:

- `episode_length`
- `return`
- `terminated_reason`
- `path_length`
- `net_displacement`
- `coverage_score`
- `stuck_ratio`
- `collision_ratio`
- `goal_reached`
- `mean_speed`
- `mean_turn_rate`

These summaries should support later filtering without destroying raw coverage prematurely.

The implementation should prefer “record and flag” over “record only if already judged good”.

## Integration Strategy

The generator should reuse the existing environment code under `envs/`, but it should not be built as a thin extension of the local `rl-baselines3-zoo/` training runtime.

Reasons:

- the target policy is not an SB3 checkpoint policy
- dataset generation should not require training artifacts
- the generator should remain conceptually separate from training and evaluation utilities

The practical implication is:

- keep environment definitions in `envs/`
- create a dedicated dataset-generation entrypoint outside the training stack
- reuse ideas or helper code from existing recording scripts only when useful

## Success Criteria

The design is successful if the first implementation can:

- generate `PointMaze` episodes with a weak random-control policy
- write a shard-based `Zarr` dataset
- preserve full raw rollout fields needed for future relabeling
- write stable spatial annotations
- produce reproducible data regardless of worker count
- attach episode quality summaries for later filtering

## Open Decisions Resolved

The following design choices are explicitly fixed for the first version:

- environment scope: `PointMaze` only
- data layers: rollout plus lightweight spatial annotation only
- split generation: deferred
- fixed-window export: deferred
- storage organization: shard-based
- target long-term format: `Zarr`
- rollout policy family: weak random-control policy
- annotation scope: stable low-risk spatial fields only

## Scope Check

This design is narrow enough to be covered by a single implementation plan.

It has one environment, one policy family, one storage strategy, and a constrained annotation set. Additional environments, richer semantic labeling, and downstream dataset products should be treated as later extensions rather than part of this first implementation.
