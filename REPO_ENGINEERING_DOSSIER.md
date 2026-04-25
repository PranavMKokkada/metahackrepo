# CodeOrganismVM / Autonomous SRE Control Center — Engineering Dossier

## 1) What this repository is

This repo implements a **hostile RL environment** where an LLM agent must maintain system health under continuous fault injection.  
At code level, it is still fundamentally **CodeOrganismVM** (vitality, faults, checkpoints), while product framing has pivoted to an **Autonomous SRE Control Center** (SLA, downtime saved, chaos incidents).

Core stack:
- **Backend/API**: FastAPI (`app.py`)
- **Environment engine**: `environment.py`
- **Simulation + faults + tests**: `data.py`
- **Typed schemas**: `models.py` (Pydantic)
- **Reward composition**: `rubrics.py`
- **UI**: Gradio (`ui.py`)
- **Training scaffolding**: `training/` + `gym_wrapper.py`

---

## 2) High-level architecture

### Runtime flow
1. Client calls `POST /reset` with a phase (`phase_1/2/3`).
2. `CodeOrganismEnv.reset()` builds a seeded synthetic codebase and injects initial faults.
3. Agent sends structured `Action` objects to `POST /step`.
4. Environment applies:
   - watchdog checks
   - vitality cost
   - action effects (patch/quarantine/rollback/subagent/expert/signal)
   - periodic/adversarial fault injection
   - test execution in temp sandbox
   - reward computation (R1–R5 + watchdog)
5. Terminal states:
   - vitality reaches 0 (death)
   - thriving condition (3 all-pass steps + vitality > 80)
   - max step timeout

### Main modules and responsibilities

| File | Purpose |
|---|---|
| `app.py` | OpenEnv-style API (`/reset`, `/step`, `/state`, `/tasks`, `/grader`) + session APIs + MCP tool endpoints + Gradio mount at `/ui` |
| `environment.py` | Episode lifecycle, costs, phase config, watchdog, action handlers, reward orchestration, metrics |
| `data.py` | Procedural codebase generation, fault catalogs (phase-aware), test execution sandbox, checkpoints/rollback, dependency graph |
| `models.py` | Typed models for Observation, Action, RewardBreakdown, StepResult, EnvState, etc. |
| `rubrics.py` | Composable reward implementation for R1–R5 |
| `tasks.py` | Curriculum task definitions + grader replay |
| `ui.py` | Dashboard UX around reset/step/chaos triggers and telemetry |
| `training/generate_sft_data.py` | Synthetic SFT trace generation |
| `training/grpo_train.py` | GRPO scaffold (placeholder-style starter, not full train pipeline) |
| `validate.py` | API compliance pre-submission checks |
| `openenv.yaml` | OpenEnv manifest metadata, actions, rubric weights |

---

## 3) Environment mechanics in detail

### Action schema and costs
Defined in `models.py` and `environment.py` (`VITALITY_COSTS`):
- `patch_file`: -2
- `run_tests`: -3
- `spawn_subagent`: -5
- `quarantine`: -1
- `rollback`: -4
- `request_expert`: -6
- `emit_signal`: 0
- `do_nothing`: 0

### Phase progression
`PHASE_CONFIG` in `environment.py`:
- `phase_1`: 20 steps, fault interval 8, initial faults 1
- `phase_2`: 50 steps, fault interval 6, initial faults 3
- `phase_3`: 100 steps, fault interval 4, initial faults 4 + targeted/adaptive behavior

### Fault model
`data.py` includes 12 fault classes across phases:
- Phase 1: `corrupted_import`, `flipped_assertion`, `missing_env_var`, `null_return`, `off_by_one`
- Phase 2 adds: `dependency_cycle`, `permission_revoked`, `race_condition`, `schema_mismatch`
- Phase 3 adds: `targeted_regression`, `cascade_corruption`, `checkpoint_invalidation`

### Reward model
`rubrics.py` computes weighted sum:
- R1 vitality delta: 35%
- R2 test recovery: 30%
- R3 efficiency (1/sqrt(n)): 15%
- R4 coordination/intent: 10%
- R5 generalization/held-out: 10%
- watchdog penalties added as negative offset

### Security/watchdog behavior
- Protected-path detection (`data.is_protected_path`) blocks/penalizes edits to tests/system-like paths.
- Path traversal and absolute-path patterns are treated as protected.
- Watchdog penalties currently implemented primarily around protected patch attempts.

### Sandbox execution
- Tests are materialized into temporary files and run via subprocess Python in isolated temp dir.
- Per-test timeout and environment variable injection are present.
- Quarantined modules are excluded and produce test errors.

---

## 4) API & interoperability surface

### Environment endpoints
- `GET /`, `GET /health`, `GET /metadata`, `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`

### Session endpoints
- `POST /sessions/create`
- `DELETE /sessions/{session_id}`
- `GET /sessions`

### MCP-style endpoints
- `GET /tools/list`
- `POST /tools/call`

Note: MCP tool-call implementation currently references `CodeOrganismActionType` in `app.py` without importing it, which is a probable runtime issue for `/tools/call`.

---

## 5) Training and evaluation maturity

### Present
- Gym wrapper (`gym_wrapper.py`) for RL integration.
- SFT synthetic data generation script.
- GRPO script skeleton with TRL API structure.
- Validation and environment unit tests (`test_env.py`).

### Incomplete / partial
- `training/grpo_train.py` is mostly scaffolded (core training calls commented out).
- No committed reward/learning curves or experiment artifacts in repo root/docs.
- No standardized experiment tracking bundle (e.g., run configs + reproducible outputs).

---

## 6) Hackathon alignment (from “OpenEnv Hackathon Opening Ceremony” guidance)

### A) Judging criteria alignment

| Criterion | Weight | Alignment |
|---|---:|---|
| Environment Innovation | 40% | **Strong**: adversarial corruption, multi-phase faults, coordination, rollback/quarantine dynamics |
| Storytelling & Presentation | 30% | **Strong**: polished UI + clear narrative framing as autonomous SRE |
| Showing Reward Improvement | 20% | **Weak/Partial**: no strong in-repo training curves/results artifacts |
| Reward & Training Pipeline | 10% | **Partial**: rubric design is good, but end-to-end GRPO training evidence is limited |

### B) Minimum submission requirements checklist

| Requirement (hackathon guidance) | Status in repo |
|---|---|
| Use OpenEnv framework & standard API | **Mostly met** (OpenEnv-style API + manifest + env mechanics) |
| Working training script (Unsloth/TRL) | **Partial** (script exists, but core train path scaffolded) |
| Evidence of real training (loss/reward plots) | **Missing/weak in repo artifacts** |
| Short writeup/video/slides linked from README | **Partial** (README + whitepaper exist, external evidence links should be strengthened) |
| Push environment to Hugging Face Space | **Claimed via metadata** (`openenv.yaml`) |
| README motivates problem, env, and results | **Strong narrative**, but empirical result evidence could be more explicit |

---

## 7) Strengths

1. **Clear environment abstraction**: clean separation of API, environment core, simulator, models.
2. **Rich fault curriculum**: phase-scaled progression and targeted adversarial faults.
3. **Composable reward design**: explicit R1–R5 implementation and weighted aggregation.
4. **Good engineer ergonomics**: typed models, reset/step/state interfaces, grader replay path.
5. **Demo-ready UI**: strong mentor/demo usability with operational framing.
6. **Security intent**: watchdog + protected-path constraints + sandboxed test execution.

---

## 8) Weaknesses and technical debt

1. **Terminology drift**: codebase mixes CodeOrganismVM and Autonomous SRE naming, causing conceptual inconsistency.
2. **Packaging inconsistency**: `pyproject.toml` metadata still says `shadow-council`, unrelated to current product.
3. **Training evidence gap**: repo lacks robust, reproducible “agent improved” artifacts.
4. **Potential runtime defect**: `CodeOrganismActionType` usage in `app.py` MCP path without visible import.
5. **Heuristic shortcuts**: some expert and subagent logic is simulation-heavy and may overstate realism.
6. **Documentation mismatch risk**: README/whitepaper claims may be ahead of fully demonstrated implementation proof.

---

## 9) Missing parts (priority order)

### P0 (must-have for judging confidence)
1. Finalize runnable RL pipeline (`training/grpo_train.py`) with real training loop enabled.
2. Commit training artifacts: reward/loss curves, baseline vs trained comparison.
3. Tighten README result section with objective metrics and direct artifact links.

### P1 (quality and reliability)
1. Resolve naming/package drift (`shadow-council` and mixed branding).
2. Fix MCP `/tools/call` action-type import/validation path.
3. Add focused tests for MCP endpoints and session lifecycle edge cases.

### P2 (product polish)
1. Improve fault realism and anti-gaming checks.
2. Add clearer experiment reproducibility docs (exact commands + seeds + hardware profile).
3. Add architecture diagram and sequence diagram for onboarding.

---

## 10) How to explain this to mentors (mentor-round narrative)

### One-line pitch
“We built an OpenEnv environment where an LLM acts as an autonomous SRE in a continuously corrupting system, and learns survival via reward-shaped remediation under adversarial chaos.”

### 60-second structure
1. **Problem**: LLMs struggle with long-horizon reliability operations under evolving failures.
2. **Environment**: multi-phase fault-injection world with strict action costs, watchdog boundaries, checkpoints, quarantine, and adversarial regressions.
3. **Learning signal**: composable rubric (stability, recovery, efficiency, coordination, generalization).
4. **Outcome**: show baseline-vs-trained behavior and why this environment can produce measurable reliability skill improvements.

### What to emphasize in Q&A
- Why this is not a static coding benchmark: dynamic failures + nontrivial trade-offs.
- Why reward is hard to game: multiple weighted dimensions + watchdog penalties.
- Why this matters: maps to real reliability/autonomous-ops behaviors.

---

## 11) Software engineer / AI-agent onboarding guide

### Start locally
```bash
pip install -r requirements.txt
python app.py
```

UI: `http://localhost:7860/ui`  
API health: `GET http://localhost:7860/health`

### Core extension points
- Add new fault types: `data.py` fault catalog + `_apply_fault` logic.
- Adjust learning pressure: `environment.py` (costs, phase config, terminal conditions).
- Tune reward incentives: `rubrics.py`.
- Expand task curriculum: `tasks.py`.
- Add endpoint/tool integration: `app.py`.

### Guardrails for contributors
- Keep `Action`/`Observation` schema compatibility (`models.py`) stable.
- Preserve reset/step/state behavior contracts for training clients.
- Add tests in `test_env.py` for any environment dynamics change.

---

## 12) Current repo reality summary

This repository is **already a strong environment and demo package** with a coherent hostile-RL concept and a polished interface.  
The key delta needed for top-tier hackathon competitiveness is to convert strong narrative and architecture into **hard training evidence and reproducibility artifacts** that clearly prove measurable agent improvement end-to-end.

