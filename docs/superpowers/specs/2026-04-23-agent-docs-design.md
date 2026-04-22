# Agent-Focused Repository Docs Design

## Summary

This design defines a concise rewrite of `CLAUDE.md` and a new `AGENTS.md` for the MorphNav repository.

The target audience is AI coding agents, not human readers. Both files should act as short operational handbooks that help an agent understand the repo, find the important code, choose the right commands, avoid common mistakes, and validate changes with minimal overhead.

The two files should remain functionally identical. They should differ only in the opening sentence so they can serve different agent ecosystems without introducing maintenance drift.

## Goals

- Give an agent a fast, accurate orientation to the repository.
- Document only the paths, workflows, and caveats that affect agent behavior.
- Replace vague or stale guidance with concrete, actionable instructions.
- Keep both files short enough to read quickly before work begins.
- Minimize future drift by keeping `CLAUDE.md` and `AGENTS.md` almost identical.

## Non-Goals

- Do not turn either file into a full human-facing project guide.
- Do not document every experiment script, flag, or historical workflow.
- Do not attempt to fully replace `README.md`, inline code comments, or source inspection.
- Do not describe implementation details that are too volatile to stay accurate.

## Target Files

- `CLAUDE.md`
- `AGENTS.md`

## Content Strategy

Both files should use the same five-section structure:

1. `Repo Summary`
2. `Key Paths`
3. `Typical Commands`
4. `Agent Rules`
5. `Gotchas`

The documents should be short, operational, and written in direct English. They should avoid narrative architecture explanations unless those details change agent decisions.

## Section Design

### 1. Repo Summary

This section should be limited to two to four sentences.

It should state that:

- MorphNav is a maze-navigation reinforcement learning research repository.
- The main local code lives in custom environments under `envs/` and auxiliary model code under `components/`.
- Training and evaluation rely on a locally customized `rl-baselines3-zoo/`.
- Some scripts are research artifacts or experiment notes rather than polished automation.

### 2. Key Paths

This section should be a short list, one line per path, covering only the paths an agent is likely to need immediately:

- `envs/` for environment definitions and registration.
- `components/` for auxiliary extractors and recurrent policy extensions.
- `scripts/` for setup and example experiment commands.
- `tests/` for smoke tests, probes, and analysis scripts.
- `rl-baselines3-zoo/` for training code, configs, and recording/extraction utilities used by this repo.

No deeper path inventory is needed unless a path is a stable entrypoint that agents repeatedly need.

### 3. Typical Commands

This section should include only a small set of representative commands:

- environment setup
- a representative training command
- log inspection with TensorBoard
- a representative recording or extraction command
- one or two lightweight test commands

Command examples should be normalized where possible:

- remove personal GPU ids
- avoid user-specific absolute paths unless unavoidable
- present scripts as starting points rather than guaranteed-safe automation

If a command is better represented as “see this script first”, that is acceptable, but the document should still name the relevant file explicitly.

### 4. Agent Rules

This section should contain short operational bullets that affect behavior in this repo.

Expected rules include:

- read existing scripts before inventing new commands
- treat `scripts/*.sh` as examples, not production-quality tooling
- avoid editing `rl-baselines3-zoo/` unless the task truly requires training-stack changes
- inspect related registration/config surfaces when changing environment behavior
- prefer minimal, local edits
- validate changes with the smallest relevant check

The rules should stay repository-specific and avoid generic coding-assistant boilerplate.

### 5. Gotchas

This section should explicitly call out repo realities that can mislead an agent:

- `README.md` is incomplete
- some shell scripts contain hard-coded GPU ids or local absolute paths
- some files under `tests/` are exploratory scripts rather than strict automated tests
- the local `rl-baselines3-zoo/` should not be assumed identical to upstream

This section should remain concise and concrete.

## File Relationship

The two files should be almost exact mirrors.

Allowed difference:

- opening sentence naming the target agent

Disallowed differences:

- separate command sets
- separate behavioral rules
- separate repo summaries

The goal is compatibility across tools without maintaining two divergent knowledge sources.

## Writing Style

The wording should follow these constraints:

- English only
- short, direct sentences
- low abstraction
- no promotional or tutorial tone
- no unnecessary architecture prose
- prefer explicit warnings to generic advice

The result should read like a compact operator handbook for coding agents.

## Success Criteria

The rewrite is successful if an agent can read either file and quickly answer:

- what this repo is for
- where the important code lives
- what commands to use as starting points
- which parts of the repo are risky or non-canonical
- how to make changes without overreaching

## Scope Check

This work remains small enough for a single implementation plan and does not need to be split further.
