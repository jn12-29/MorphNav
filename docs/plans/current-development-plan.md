# Current Development Plan

## Documentation Policy

- `README.md` is the current user entry point.
- `AGENTS.md` is the project-specific mistake-prevention note for agents.
- The old `docs/superpowers/` plugin documents are no longer the source of truth.
- `docs/ref/` is local reference material only and is not part of the tracked plan.

## Phase 1: PointMaze Ball Model Offline PI Rehearsal

Detailed plan: [`phase-1-offline-pi-rehearsal.md`](./phase-1-offline-pi-rehearsal.md).

Phase 1 makes the current MuJoCo ball model (`PointMaze`) usable for offline path-integration rehearsal and held-out probing while keeping the existing shared actor-LSTM PI head architecture.

Core commitments:

- `offline_pi_rehearsal` performs PI-only updates on the current RL model with a separate optimizer.
- `offline_pi_probe` evaluates held-out PI performance without parameter updates.
- `render_pointmaze_trajectory` provides explicit MP4/NPZ/JSON analysis renders for dataset replay, offline PI probe predictions, and recurrent-policy rollouts.
- `obs/*` dataset arrays are action-before policy observations aligned with `step/action`.
- `achieved_goal` remains in observations as the PI target; `achieved_goal` is dropped from per-step policy features, while `start_pos` stays in per-step policy features and seeds LSTM initial states through `pi_init_state_key='start_pos'`.
- `path_integration_head` and PI initial-state projection layers must be included in the online PPO optimizer parameter set.
- Existing online rollout PI behavior in `PathIntegrationRecurrentPPO.train()` is preserved except for PI optimizer membership and `start_pos` LSTM-state initialization at episode starts.
- RL interleaving should be exposed through a lightweight SB3 callback or runner hook after the standalone APIs are in place.
- Phase 1 remains limited to the PointMaze MuJoCo ball model.

## Phase 2: Separate PI/RL Recurrent Architecture

After Phase 1 is working, evaluate a new architecture where PI and RL use separate recurrent modules. The PI recurrent representation may be concatenated with other observation features and fed into the RL recurrent path. This phase should be treated as an architecture experiment, not as a prerequisite for offline PI rehearsal.

Key decisions for that phase include whether RL loss can update the PI recurrent module, whether PI representations are detached before entering the RL path, and how to avoid `achieved_goal` leakage.

## Phase 3: AntMaze / Embodied Compatibility

After PointMaze ball-model rehearsal is stable, extend the dataset and offline PI path toward embodied agents such as `AntMaze`. The PI target can remain the global xy position, but AntMaze needs an environment adapter, a suitable exploration policy for multi-dimensional actions, and policy-compatible observations for its larger observation/action spaces.

AntMaze support is intentionally deferred until the PointMaze ball-model workflow is complete.
