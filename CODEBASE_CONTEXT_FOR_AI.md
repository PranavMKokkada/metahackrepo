# Codebase Context For AI

This document is a handoff brief for any AI agent or developer picking up this repository. It explains what the codebase is, what has already been implemented, how the pieces fit together, how to run it, and what caveats matter before making changes.

## Project Summary

This repository implements **Autonomous SRE Control Center**, an OpenEnv-style reinforcement learning environment for training LLM agents to act like autonomous site reliability engineers.

The environment simulates a hostile, self-corrupting software/service system. An agent must preserve system health by diagnosing failures, patching code, rolling back bad states, quarantining broken modules, coordinating subagents, and consulting an expert oracle. The core idea is to train or evaluate LLMs on long-horizon operational repair under chaos engineering pressure.

Primary hackathon framing:

- Event: Meta/PyTorch OpenEnv Hackathon
- Theme: Self-Improvement
- Project domain: Autonomous SRE, chaos engineering, incident remediation
- Core claim: LLM agents can learn operational repair behavior through environment rewards and feedback loops

## Current State In One Sentence

The codebase currently has a functional environment, FastAPI server, Gradio UI, Gym wrapper, OpenEnv-style manifest, grader, test suite, Docker/CI setup, and synthetic training scaffolding, but it still needs real training evidence and stronger result artifacts to be a winning hackathon submission.

## Important Current Branch/Repo Notes

- The active working directory is the main project repo.
- There is a nested `.main-worktree` directory that appears as a tracked gitlink/worktree artifact in this branch. Be careful before deleting or modifying it.
- The current repo has recently passed `pytest -q` with `58 passed`.
- A separate roadmap file exists: `HACKATHON_WINNING_ROADMAP.md`.
- A prior audit exists: `audit_report.md`.

Before changing code, run:

```bash
git status --short --branch
```

There may be work from the user or previous AI sessions. Do not revert unrelated changes unless explicitly asked.

## Repository Map

| Path | Purpose |
|---|---|
| `app.py` | FastAPI server, OpenEnv-like HTTP endpoints, auth, sessions, MCP-style tool endpoints, Gradio mounting |
| `environment.py` | Core environment lifecycle: reset, step, state, reward orchestration, action handling |
| `data.py` | Procedural codebase simulator, fault injection, patch application, test execution, dependency graph |
| `models.py` | Pydantic models for actions, observations, rewards, state, files, tests, checkpoints |
| `rubrics.py` | Composable R1-R5 reward implementation |
| `tasks.py` | Phase/task definitions and grader replay logic |
| `client.py` | HTTP client wrapper for interacting with the environment |
| `baseline.py` | OpenAI-based baseline inference script |
| `inference.py` | Hugging Face/OpenAI-compatible inference runner with strict logging |
| `gym_wrapper.py` | Gymnasium-compatible wrapper for RL libraries |
| `ui.py` | Gradio “control center” UI |
| `validate.py` | Pre-submission validator against a live server |
| `openenv.yaml` | OpenEnv-style environment manifest |
| `Dockerfile` | Runtime container for HF Space/local deployment |
| `requirements.txt` | Runtime dependencies |
| `requirements-training.txt` | Training dependencies |
| `pyproject.toml` | Package metadata, optional training deps, pytest config |
| `test_env.py` | Unit tests for models, simulator, environment, rewards, sessions, grader |
| `test_api.py` | FastAPI contract tests using TestClient |
| `.github/workflows/ci.yml` | CI: install deps, syntax check, pytest, pip-audit, Docker build |
| `training/generate_sft_data.py` | Generates synthetic SFT traces from simulator internals |
| `training/grpo_train.py` | Placeholder/scaffold GRPO script, not yet a full training pipeline |
| `training/curriculum.py` | Curriculum phase advancement helper |
| `training/sft_data.jsonl` | Generated synthetic SFT data |
| `evaluation/held_out_seeds.json` | Held-out seed registry |
| `README.md` | Public-facing project explanation and quickstart |
| `TECHNICAL_WHITEPAPER.md` | Longer product/system narrative |
| `REPO_ENGINEERING_DOSSIER.md` | Engineering audit/architecture notes |
| `HACKATHON_WINNING_ROADMAP.md` | Detailed roadmap of remaining work to improve hackathon competitiveness |

## Core Concept

The environment models a service/codebase organism under active corruption. The agent receives observations and takes actions.

The agent sees:

- timestep and max steps
- vitality/SLA health score
- file tree with checksums and contents
- test results
- stack trace from failing tests
- environment variables from the simulator
- active checkpoints
- recent signals
- subagent results
- watchdog flags
- dependency graph
- alerts

The agent can:

- patch files
- run tests
- roll back to checkpoints
- spawn subagents
- quarantine modules
- request expert feedback
- emit coordination signals
- do nothing

The environment rewards:

- maintaining vitality/SLA
- recovering failing tests
- solving efficiently
- coordinating intent/subagents
- generalizing on held-out/adversarial phases
- respecting watchdog/security constraints

## Environment Phases

The environment has three phases defined in `environment.py` and `tasks.py`.

| Phase | Max Steps | Fault Interval | Initial Faults | Purpose |
|---|---:|---:|---:|---|
| `phase_1` | 20 | 8 | 1 | Basic single-fault diagnosis and repair |
| `phase_2` | 50 | 6 | 3 | Multi-fault survival and coordination |
| `phase_3` | 100 | 4 | 4 | Adversarial faults targeting recent repairs |

Fault progression:

- Phase 1 faults:
  - corrupted import
  - flipped assertion
  - missing env var
  - null return
  - off-by-one
- Phase 2 adds:
  - dependency cycle
  - permission revoked
  - race condition
  - schema mismatch
- Phase 3 adds:
  - targeted regression
  - cascade corruption
  - checkpoint invalidation

## Core API Surface

The FastAPI app is in `app.py`.

Public metadata endpoints:

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /schema`
- `GET /tasks`

Mutable/core endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /grader`

Session endpoints:

- `POST /sessions/create`
- `GET /sessions`
- `DELETE /sessions/{session_id}`

MCP-style endpoints:

- `GET /tools/list`
- `POST /tools/call`

By default, mutable endpoints require an API key. If no `CODEORGANISM_API_KEYS` is configured, the app generates a runtime API key for local process use. Tests import `RUNTIME_API_KEY` from `app.py`.

Useful env vars:

```bash
CODEORGANISM_API_KEYS=change-me
CODEORGANISM_AUTH_DISABLED=false
CODEORGANISM_CORS_ORIGINS=http://localhost:7860,http://127.0.0.1:7860
CODEORGANISM_RATE_LIMIT_WINDOW=60
CODEORGANISM_RATE_LIMIT_MAX=120
CODEORGANISM_MAX_SESSIONS=64
CODEORGANISM_SESSION_TTL_SECONDS=3600
PORT=7860
HOST=127.0.0.1
```

## Actions

Actions are defined by `CodeOrganismActionType` in `models.py`.

| Action | Payload | Purpose |
|---|---|---|
| `patch_file` | `path`, `diff` | Apply constrained `OLD|NEW` patch to a simulated file |
| `run_tests` | optional `test_suite` | Execute all simulator tests |
| `spawn_subagent` | `task`, optional `context` | Simulate delegated repair |
| `quarantine` | `module` or `path` fallback | Isolate corrupt module |
| `rollback` | `checkpoint_id` | Restore simulator state |
| `request_expert` | `query` | Ask expert/heuristic/LLM validator for patch quality |
| `emit_signal` | `signal_type`, `signal_data` | Broadcast intent or metadata |
| `do_nothing` | none | No-op/monitoring action |

Patch format:

```text
OLD_TEXT|NEW_TEXT
```

The simulator does not accept arbitrary full-file writes in the current hardened path. Patches are constrained and Python code is AST-validated.

## Reward System

Reward breakdown is represented by `RewardBreakdown` in `models.py`.

| Reward | Weight | Meaning |
|---|---:|---|
| R1 `vitality_delta` | 35% | Change in vitality/SLA |
| R2 `test_recovery` | 30% | FAIL to PASS improvements, PASS to FAIL penalties |
| R3 `efficiency_bonus` | 15% | Fewer actions are better; duplicate actions penalized |
| R4 `coordination_bonus` | 10% | Intent-action alignment and coordination |
| R5 `novelty_bonus` | 10% | Held-out/adversarial generalization |
| `watchdog_penalty` | hard penalty | Security/policy violations |

Implementation is in `rubrics.py`, called from `environment.py`.

Important caveat:

The reward system exists and is test-covered, but it still needs stronger anti-gaming logic for a winning submission. Repeated no-op or signal behavior should be made less rewarding in future work.

## Simulator Details

`data.py` owns the in-memory simulated codebase.

It currently provides:

- procedural module selection
- generated unit tests
- deterministic seed-based generation
- in-memory file tree
- env var simulation
- fault injection
- checkpoint creation and rollback
- quarantine mechanics
- constrained patch application
- safe-ish AST validation
- in-process test execution
- dependency graph extraction
- expert patch quality evaluation

Important caveat:

The simulator is currently a convincing prototype, but many generated modules are generic utility-code modules. For a stronger SRE story, future work should make modules and incidents feel more like real services: API gateway, auth service, queue worker, cache layer, deployment controller, latency SLOs, logs, alerts, and blast radius.

## Training State

Training-related files exist, but this is the weakest part of the project.

Implemented:

- `training/generate_sft_data.py` creates synthetic SFT examples.
- `training/sft_data.jsonl` exists.
- `training/curriculum.py` tracks phase advancement gates.
- `training/grpo_train.py` exists as a GRPO scaffold.
- `CodeOrganismVM_Training.ipynb` contains a notebook-style training flow.
- `gym_wrapper.py` exposes a discrete Gymnasium interface.

Not yet truly complete:

- `training/grpo_train.py` does not perform full environment-connected GRPO.
- Current reward callbacks in the scaffold are placeholder-like.
- The notebook contains mock reward curve plotting.
- There are no committed real training plots.
- There is no committed baseline-vs-trained evaluation summary.
- There is no linked trained model artifact.

This is the main thing the next AI should improve if the goal is to win the hackathon.

## Testing And CI

Current local verification:

```bash
pytest -q
```

Recent result:

```text
58 passed
```

CI does:

- checkout
- Python setup
- install `requirements.txt`
- install `pip-audit`
- syntax check key files
- run `pytest -q`
- run `pip-audit -r requirements.txt`
- run Docker build

Important tests:

- `test_env.py`: model contracts, simulator, fault injection, reward behavior, sessions, grader
- `test_api.py`: FastAPI TestClient contract tests

Before major changes, run:

```bash
python -m py_compile app.py data.py environment.py gym_wrapper.py inference.py models.py tasks.py ui.py validate.py baseline.py client.py training/generate_sft_data.py training/grpo_train.py training/curriculum.py test_api.py test_env.py
pytest -q
```

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API key:

```bash
set CODEORGANISM_API_KEYS=change-me
```

On Unix-like shells:

```bash
export CODEORGANISM_API_KEYS=change-me
```

Start server:

```bash
python app.py
```

Open UI:

```text
http://localhost:7860/ui
```

Run live validator:

```bash
python validate.py --api-url http://localhost:7860 --api-key change-me
```

Run a client interaction:

```python
from client import SREEnvClient
from models import Action, CodeOrganismActionType

client = SREEnvClient("http://localhost:7860", api_key="change-me")
obs = client.reset("phase_1")
result = client.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="heartbeat"))
print(result.reward)
```

## Docker

Build:

```bash
docker build -t autonomous-sre .
```

Run:

```bash
docker run -p 7860:7860 -e CODEORGANISM_API_KEYS=change-me autonomous-sre
```

The Dockerfile:

- uses `python:3.11-slim`
- installs runtime requirements
- copies selected source files
- runs `pytest -q` during build
- runs as non-root `appuser`
- exposes port `7860`
- has a healthcheck against `/health`

## Hugging Face Space

`openenv.yaml` currently points to:

```text
https://huggingface.co/spaces/PranavMKokkada/autonomous-sre
```

Before submission, verify:

- Space exists.
- Space builds.
- `/health` works.
- UI loads.
- `/tasks` works.
- `/reset` and `/step` can be used in judge/demo mode.
- README links to the Space.

## What Has Been Accomplished

### Environment Core

- Implemented stateful environment lifecycle.
- Implemented `reset`, `step`, and `state`.
- Implemented task phases.
- Implemented vitality/SLA health.
- Implemented termination conditions:
  - organism death
  - thrival
  - timeout
- Implemented action cost model.
- Implemented auto-checkpointing.
- Implemented rollback limits.
- Implemented quarantine.
- Implemented subagent simulation.
- Implemented expert/patch quality evaluation fallback.
- Implemented dependency graph observations.

### Fault Simulation

- Implemented 12 fault types across 3 phases.
- Implemented random and targeted fault injection.
- Implemented cascade/checkpoint/adversarial phase concepts.
- Implemented held-out seed registry hook.

### API And Integration

- Implemented FastAPI server.
- Implemented OpenEnv-like endpoints.
- Implemented MCP-style tools list/call.
- Implemented session management with TTL/max-session support.
- Implemented auth and rate limiting.
- Implemented CORS configuration.
- Implemented client wrapper.
- Implemented grader endpoint.

### UI

- Implemented Gradio dashboard.
- Mounted UI at `/ui`.
- UI can inspect state and take actions.
- UI includes SRE/control-center framing.

### Tests And DevOps

- Implemented comprehensive unit tests.
- Implemented API tests.
- Added CI workflow.
- Added Dockerfile.
- Split runtime and training requirements.
- Pinned runtime dependencies.
- Added dependency audit in CI.

### Documentation

- README exists.
- Technical whitepaper exists.
- Engineering dossier exists.
- Audit report exists.
- Winning roadmap exists.
- This context handoff now exists.

## Known Limitations And Caveats

### Training Evidence Is Not Complete

The biggest limitation is lack of real training evidence. Do not claim the model is GRPO-trained or has achieved a specific MTTR reduction unless you generate and commit proof.

Needed artifacts:

- real rollout logs
- baseline comparison
- reward/loss plots
- evaluation summary
- trained model or LoRA link, if applicable

### README May Overclaim

The README has historically used strong claims like “60% reduction in MTTR.” Make sure all such claims are backed by actual result artifacts before final submission.

### Reward Can Be Improved

The reward system is implemented, but should be hardened:

- penalize repeated no-op/signal loops
- reward actual root-cause repair
- strengthen R4 coordination
- make R5 held-out generalization measurable

### SRE Realism Can Be Improved

The story is SRE, but some generated code is generic utility code. Future work should make observations and modules more incident-like.

### OpenEnv Base Class

`environment.py` tries to import `openenv_core.Environment`, but falls back to a local stub. Confirm whether the actual submission needs stricter inheritance or framework integration.

### Public Auth/Space Mode

Mutable endpoints require API keys by default. For a public HF Space demo, decide how judges should interact:

- public demo mode with `CODEORGANISM_AUTH_DISABLED=true`
- or visible demo API key
- or UI-only interactions that inject the key internally

Do not accidentally expose private API keys.

## Recommended Next Tasks

If the next AI is coding for hackathon competitiveness, prioritize in this order:

1. Add real evaluation scripts for random, no-op, heuristic, and LLM policies.
2. Generate real rollout logs and plots.
3. Update README with honest measured results.
4. Remove unsupported claims.
5. Verify HF Space deployment.
6. Add a short demo video/blog/slides link.
7. Strengthen reward anti-gaming.
8. Improve SRE realism in generated modules and observations.
9. Add postmortem/episode summary output.
10. Add held-out seed evaluation report.

## Suggested New Files To Add

```text
training/evaluate_policy.py
training/plot_results.py
training/rollout.py
results/README.md
results/eval_summary.json
results/random_rollouts.jsonl
results/heuristic_rollouts.jsonl
results/reward_curve.png
results/baseline_vs_agent.png
results/survival_by_phase.png
```

## Suggested Evaluation Metrics

Track these per policy:

- survival rate
- thrival rate
- mean reward
- median reward
- mean final vitality
- tests recovered per episode
- watchdog violations
- mean steps per episode
- phase-specific score
- held-out seed score

Recommended result table:

| Policy | Phase 1 Survival | Phase 2 Survival | Phase 3 Survival | Mean Reward | Final Vitality | Watchdog Violations |
|---|---:|---:|---:|---:|---:|---:|
| No-op | | | | | | |
| Random | | | | | | |
| Heuristic | | | | | | |
| Trained/SFT/GRPO | | | | | | |

## Coding Guidelines For Future AI

- Keep changes scoped.
- Preserve existing tests.
- Add tests for behavior changes.
- Avoid editing generated data unless needed.
- Do not introduce fake metrics.
- Do not hardcode plot data.
- Do not claim training success without result artifacts.
- Prefer real environment rollouts over static examples.
- Keep README honest and judge-friendly.
- Run `pytest -q` before finalizing.

## Final Mental Model

Think of this repo as three layers:

1. **Environment layer:** mostly implemented and test-covered.
2. **Training/evaluation layer:** scaffolded but not yet strong.
3. **Submission/story layer:** compelling but needs evidence and cleanup.

The fastest path to a stronger submission is to build layer 2 and make layer 3 honest.
