# The bruised codebase

**Autonomous SRE / CodeOrganismVM** · OpenEnv-style incident sandbox

> Green dashboards can still feel wrong. Incidents are long stories, not single diffs.

---

## TL;DR

| You want… | We built… |
|-----------|-----------|
| A place LLMs **train**, not chat | Hostile codebase simulator + **reset / step / state** API |
| Themes: **self-improvement**, **multi-agent**, **world model** | Vitality + faults + **signals / subagents / graph / memory** |
| Proof, not vibes | **Plots + JSON** in repo, **Unsloth / TRL** notebook, **baselines** vs trained |
| A story, not a spec | This page + Space UI + README links |

---

## Hackathon themes → our angle

| Theme (Opening ceremony) | How it shows up here |
|--------------------------|----------------------|
| **Self-improvement** | Agent must recover under **new** faults; no single frozen cheat sheet wins. |
| **Long-horizon planning** | Episodes span many steps; panic patching costs vitality. |
| **Multi-agent** | **Signals**, **subagents**, expert channel: coordination before blind patches. |
| **World modelling** | **Dependency graph**, alerts, platform memory: symptom vs cause diverge. |

---

## What the agent sees, does, earns

| **Sees** | **Does** | **Earns (rubric slice)** |
|----------|----------|---------------------------|
| Tests, files, vitality | `patch_file`, `run_tests`, `rollback`, `quarantine`, … | Vitality delta, test recovery |
| Stack traces, graph | `emit_signal`, `spawn_subagent`, `request_expert` | Efficiency, coordination, novelty |
| Watchdog boundaries | Stops unsafe paths | Penalties if it games protected zones |

**Design goal:** signal is **dense**, **composable** (R1 to R5 style split), and **annoying to game** without doing real repair.

---

## Model stack (plain English)

| Stage | Role |
|-------|------|
| **Pretrain** | Broad language priors (we do not redo this here). |
| **SFT** | Task-shaped behaviour from demonstrations / notebook path. |
| **Preferences / RLHF** | Teaches “what humans prefer” before hard env RL. |
| **RL + env** | **This project:** policy meets **real** `step()` outcomes, long-horizon credit. |
| **Harness** | Tools, memory, API, guardrails, observability so learning survives contact with “prod shaped” UI. |

---

## Judging lens (how we lined ourselves up)

| Criterion | Weight | Our one-line answer |
|-----------|--------|---------------------|
| Environment innovation | **40%** | Self-corrupting **service codebase** + SRE ops layer, not another grid world. |
| Storytelling & presentation | **30%** | Space + blog + README: **problem → env → results → why**. |
| Improvement in rewards | **20%** | Curves + tables in `results/`, **vs** noop / random / heuristic / SFT-style baselines. |
| Reward & training pipeline | **10%** | Rubric-shaped rewards; **Unsloth / TRL** notebook tied to same env code path. |

---

## Four beats (the storyboard)

| # | Beat | In one sentence |
|---|------|------------------|
| 1 | **Problem** | LLMs need **long** incident practice, not one-shot trivia. |
| 2 | **Environment** | Typed world: faults, tests, vitality, graph, platform telemetry, standard OpenEnv surface. |
| 3 | **Results** | Show **before/after** and **vs baseline** with committed plots, not orphan Colab-only charts. |
| 4 | **Why it matters** | If we learn here, we learn how to push agents toward **messier** real systems later. |

---

## Where everything lives

| Artifact | Link / path |
|----------|-------------|
| **Live demo** | This Hugging Face **Space** (Docker: API + Gradio + console) |
| **Training notebook** | `CodeOrganismVM_Training.ipynb` (Unsloth + TRL oriented) |
| **Numbers & plots** | GitHub: `results/`, scripts under `training/` |
| **Upstream OpenEnv** | [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) |
| **Manifest** | `openenv.yaml` in repo root |

---

## Table stakes (boring but real)

| Rule | Why |
|------|-----|
| **Client / server** split | API clients do not import server-only hacks. |
| **`openenv.yaml` + `reset` / `step` / `state`** | Others can **actually** hook trainers to the same contract. |
| **Docker = Space** | If it does not ship, it does not count. |

---

## Closing

We want a model to sit with a **bruised codebase** until it stops flinching. That needs **code**, **curves**, and a **short** story. Thanks for reading.

*Team Autonomous SRE*
