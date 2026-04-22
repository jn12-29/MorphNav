# Agent Docs Rewrite Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** Rewrite `CLAUDE.md` and add `AGENTS.md` as concise, agent-first operator handbooks for this repository.

**架构：** Keep both files nearly identical and drive them from one validated content shape: repo summary, key paths, representative commands, agent rules, and gotchas. Reuse concrete facts from the current repository state, remove stale or personal details where possible, and keep the documents short enough for quick agent onboarding.

**技术栈：** Markdown, git, local repository inspection

---

## File Structure

- Modify: `CLAUDE.md`
  Responsibility: replace the current ad hoc guidance with the approved five-section agent handbook.
- Create: `AGENTS.md`
  Responsibility: mirror `CLAUDE.md` with only the opening sentence adjusted for generic coding agents.
- Reference: `README.md`
  Responsibility: source for current high-level repo framing, but not treated as canonical.
- Reference: `envs/__init__.py`
  Responsibility: confirm registered environments and stable environment names.
- Reference: `components/aux_policy.py`
  Responsibility: confirm local auxiliary-policy customization.
- Reference: `rl-baselines3-zoo/conf/maze.yml`
  Responsibility: confirm training stack defaults and representative training entrypoint.
- Reference: `scripts/build_conda_env.sh`
  Responsibility: inspect current setup workflow and normalize it into agent-safe guidance.
- Reference: `scripts/sb3zoo_train.sh`
  Responsibility: inspect current training workflow and select a representative command.
- Reference: `scripts/sb3_extract_infos.sh`
  Responsibility: inspect current extraction workflow and select a representative command.

### Task 1: Finalize canonical content for the short handbook

**Files:**
- Modify: `CLAUDE.md`
- Create: `AGENTS.md`
- Reference: `README.md`
- Reference: `envs/__init__.py`
- Reference: `components/aux_policy.py`
- Reference: `rl-baselines3-zoo/conf/maze.yml`
- Reference: `scripts/build_conda_env.sh`
- Reference: `scripts/sb3zoo_train.sh`
- Reference: `scripts/sb3_extract_infos.sh`

- [ ] **Step 1: Re-read the source files that define repo facts**

运行：`sed -n '1,220p' README.md && sed -n '1,220p' envs/__init__.py && sed -n '1,220p' rl-baselines3-zoo/conf/maze.yml`
预期：Recover the stable repo summary, registered env ids, and current training defaults.

- [ ] **Step 2: Extract the minimal content that belongs in both docs**

```md
Repo Summary
- MorphNav is a maze-navigation RL research repo.
- Core local code lives in envs/ and components/.
- Training uses the local rl-baselines3-zoo/ tree.

Key Paths
- envs/
- components/
- scripts/
- tests/
- rl-baselines3-zoo/

Typical Commands
- setup
- train
- tensorboard
- record/extract
- quick smoke tests

Agent Rules
- read scripts before inventing new commands
- treat scripts as examples
- avoid changing rl-baselines3-zoo/ unless needed
- validate with the smallest relevant check

Gotchas
- README is incomplete
- scripts may contain hard-coded GPU ids and local paths
- some tests are exploratory scripts
```

- [ ] **Step 3: Normalize command examples into agent-safe forms**

```md
Setup:
conda create -n mz python=3.12 -y
conda run -n mz pip install -r requirements.txt
pip install -e ./rl-baselines3-zoo

Train:
python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml ...

Inspect logs:
tensorboard --logdir logs/

Extract:
python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs ...

Smoke tests:
python tests/point_maze_env_test.py
python tests/ant_maze_env_test.py
```

- [ ] **Step 4: Check the final content against the approved design spec**

运行：`sed -n '1,240p' docs/superpowers/specs/2026-04-23-agent-docs-design.md`
预期：Confirm the content still matches the five-section structure and the “almost identical files” rule.

- [ ] **Step 5: Commit the planning checkpoint**

```bash
git add docs/superpowers/plans/2026-04-23-agent-docs-implementation.md
git commit -m "docs: add agent docs implementation plan"
```

### Task 2: Rewrite `CLAUDE.md` as the canonical short handbook

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace the current content with the approved five-section structure**

```md
# CLAUDE.md

This file guides Claude Code when working in this repository.

## Repo Summary
...

## Key Paths
...

## Typical Commands
...

## Agent Rules
...

## Gotchas
...
```

- [ ] **Step 2: Fill the summary and path sections with concrete repository facts**

```md
## Repo Summary

MorphNav is a maze-navigation reinforcement learning research repository.
Most local project logic lives in custom environments under `envs/` and auxiliary policy code under `components/`.
Training and evaluation use the local customized `rl-baselines3-zoo/` tree.
Treat many shell scripts as experiment notes or starting points, not polished automation.

## Key Paths

- `envs/`: custom Gymnasium/MuJoCo environments and environment registration.
- `components/`: auxiliary recurrent-policy and feature-extractor code.
- `scripts/`: setup and example experiment commands.
- `tests/`: smoke tests, probes, and analysis scripts.
- `rl-baselines3-zoo/`: local training stack, configs, and recording utilities used by this repo.
```

- [ ] **Step 3: Fill the command, rules, and gotcha sections**

```md
## Typical Commands

Use these as starting points and inspect the related script before running variants.

```bash
conda create -n mz python=3.12 -y
conda run -n mz pip install -r requirements.txt
pip install -e ./rl-baselines3-zoo
python ./rl-baselines3-zoo/train.py --algo ppo_lstm --env PointMaze -conf ./rl-baselines3-zoo/conf/maze.yml
tensorboard --logdir logs/
python ./rl-baselines3-zoo/rl_zoo3/record_video_with_data.py --algo ppo_lstm --env PointMaze -f ./logs --exp-id <ID> -n 1000 --load-best
python tests/point_maze_env_test.py
python tests/ant_maze_env_test.py
```

## Agent Rules

- Read `scripts/` before inventing new training or extraction commands.
- Treat `scripts/*.sh` as examples; some contain hard-coded devices or paths.
- Avoid editing `rl-baselines3-zoo/` unless the task really requires training-stack changes.
- When changing env behavior, inspect `envs/__init__.py`, the env constructor, and any dependent scripts together.
- Keep edits minimal and run the smallest relevant validation.

## Gotchas

- `README.md` is incomplete and should not be treated as the full source of truth.
- Some shell scripts contain local absolute paths and fixed GPU ids.
- Some files under `tests/` are exploratory analysis scripts, not strict automated tests.
- The local `rl-baselines3-zoo/` may differ from upstream behavior and APIs.
```

- [ ] **Step 4: Review the final file for brevity and duplication**

运行：`sed -n '1,220p' CLAUDE.md`
预期：The file is short, direct, and stays within the approved five-section structure.

- [ ] **Step 5: Commit the rewritten Claude handbook**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE handbook for agents"
```

### Task 3: Create `AGENTS.md` as the mirrored generic-agent handbook

**Files:**
- Create: `AGENTS.md`
- Reference: `CLAUDE.md`

- [ ] **Step 1: Copy the validated `CLAUDE.md` structure into a new file**

```md
# AGENTS.md

This file guides coding agents when working in this repository.

## Repo Summary
...
```

- [ ] **Step 2: Keep all sections identical except the opening sentence**

```md
Difference allowed:
- opening sentence only

No other wording drift:
- same repo summary
- same key paths
- same commands
- same rules
- same gotchas
```

- [ ] **Step 3: Compare the two files directly**

运行：`diff -u <(tail -n +3 CLAUDE.md) <(tail -n +3 AGENTS.md)`
预期：No meaningful differences after the opening sentence.

- [ ] **Step 4: Review the new file**

运行：`sed -n '1,220p' AGENTS.md`
预期：The file matches `CLAUDE.md` in structure and operational content.

- [ ] **Step 5: Commit the mirrored agent handbook**

```bash
git add AGENTS.md
git commit -m "docs: add AGENTS handbook"
```

### Task 4: Final verification and handoff

**Files:**
- Modify: `CLAUDE.md`
- Create: `AGENTS.md`

- [ ] **Step 1: Run a docs-focused final diff review**

运行：`git diff -- CLAUDE.md AGENTS.md`
预期：Only the intended handbook content changes are present.

- [ ] **Step 2: Verify the two docs satisfy the design spec**

运行：`sed -n '1,240p' docs/superpowers/specs/2026-04-23-agent-docs-design.md && sed -n '1,220p' CLAUDE.md && sed -n '1,220p' AGENTS.md`
预期：Both files match the five-section structure and the mirrored-file rule.

- [ ] **Step 3: Check repository status for unrelated user changes**

运行：`git status --short`
预期：Only the intended doc files plus any pre-existing unrelated changes remain.

- [ ] **Step 4: Commit the final integrated docs change**

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: align agent handbooks"
```

- [ ] **Step 5: Prepare final execution notes**

```md
Report:
- rewritten `CLAUDE.md`
- added `AGENTS.md`
- kept both files operational and nearly identical
- preserved unrelated user changes
```

## Self-Check

- Spec coverage: the plan covers the full rewrite of `CLAUDE.md`, creation of `AGENTS.md`, mirrored wording rules, and final verification against the approved design spec.
- Placeholder scan: no TODO/TBD markers remain.
- Consistency check: all tasks enforce the same five-section structure and the same “opening sentence only” difference rule.
