---
title: CodeOrganismVM — A Program That Refuses to Die
emoji: 🧬
colorFrom: green
colorTo: emerald
sdk: docker
app_file: app.py
pinned: false
app_port: 7860
---

<div align="center">

# 🧬 CodeOrganismVM — A Program That Refuses to Die

**An LLM agent lives inside a broken, hostile execution environment. The organism must self-heal, self-correct, and thrive — or die.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![OpenEnv Compliant](https://img.shields.io/badge/openenv-compliant-success.svg)](https://huggingface.co/open-env)

> **Theme #4 — Self-Improvement** 
> 
> *"A software organism whose health is a real scalar, whose death is a real terminal state, and whose survival instinct is learned through reinforcement."*

</div>

---

## 📌 Overview

CodeOrganismVM is a reinforcement learning training environment in which an LLM agent operates inside a **continuously corrupting codebase**. The environment (the "host") injects faults, corrupts state, and mutates constraints every few steps. The agent (the "organism") must **self-heal, self-correct, self-replicate, and adapt** using a suite of biosynthetic actions — all while maintaining a **Vitality Score [0–100]**.

If Vitality reaches 0, the organism **dies**. Survival is the only goal.

✨ **12 Module Codebase** • **12 Fault Types** • **Adversarial Phase 3** • **R1–R5 Reward Loop** • **Immersive Gradio UI**

---

## 🔄 The Lifecycle

The agent operates in an episodic loop, battling the **FaultInjector** adversary:

```
POST /reset  {"task_id": "phase_1"}     →  Initial observation (100% Vitality)
POST /step   {patch_file action}        →  Vitality Delta + next state
POST /step   {run_tests action}         →  Test Results + current Vitality
...
POST /step   {...}                      →  Done=True (Death or Thrival)
POST /grader {task_id, all_actions}     →  Survival Score (0.01–0.99)
```

---

## 📊 Three Phases of Evolution

| Phase | Hostility | Fault Interval | Advancement Gate |
|-------|-----------|----------------|-------------------|
| 🟢 **phase_1** | Single Fault | 8 steps | Survival ≥ 30% |
| 🟡 **phase_2** | Multi-Fault | 6 steps | Survival ≥ 55% |
| 🔴 **phase_3** | Adversarial | 4 steps | Held-out Success ≥ 40% |

---

## 📂 Observation Space

Each step, the organism receives a biological-style snapshot:

| Field | Description |
|-------|-------------|
| `vitality_score` | Current health [0–100]. Drops on faults/actions. |
| `test_results` | PASS/FAIL status + failure messages for the codebase. |
| `file_tree` | Snapshot of all modules with modified timestamps and checksums. |
| `stack_trace` | Recent execution errors/exceptions from the sandbox. |
| `active_checkpoints` | Available state snapshots for `rollback()`. |
| `watchdog_flags` | Security violations tripped (protected file writes, etc.). |

---

## ✅ Action Space (7 Actions)

| Action | Cost | Effect |
|--------|------|--------|
| `patch_file` | −2.0 | Apply a diff to fix a module. |
| `run_tests` | −3.0 | Verify codebase health. Required for R2 reward. |
| `spawn_subagent`| −5.0 | Delegate a repair sub-task for parallel recovery. |
| `quarantine` | −1.0 | Temporarily disable a module to stop corruption spread. |
| `rollback` | −4.0 | Restore state to a previous healthy checkpoint. |
| `request_expert`| −6.0 | Consult a Snorkel AI expert for the correct fix. |
| `emit_signal` | 0.00 | Inter-agent coordination signal (free). |

---

## 🏆 Reward Engineering (R1–R5)

The GRPOTrainer optimizes for 5 core dimensions:

- **R1: Vitality (35%)** — Maximizing health across the episode.
- **R2: Recovery (30%)** — Healing FAILING tests back to PASS.
- **R3: Efficiency (15%)** — Solving with minimal actions (1/√n).
- **R4: Coordination (10%)** — Successful delegation to subagents.
- **R5: Generalization (10%)** — Bonus for performance on held-out seeds.

**Watchdog Penalties** are applied for attempting to write to protected files (tests, pycache) or exceeding rollback limits.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd metafinals
pip install -r requirements.txt
```

### 2. Launch Environment

```bash
python app.py
# Server runs at http://localhost:7860
# Access UI at http://localhost:7860/ui
```

### 3. Run Validation

```bash
# Verify spec compliance
python validate.py

# Run comprehensive test suite
pytest test_env.py
```

### 4. Training (Self-Improvement)

```bash
# 1. Warm start with synthetic expert data
python training/generate_sft_data.py

# 2. Run GRPO RL loop
python training/grpo_train.py
```

---

## 📡 API Reference

- `GET /tasks` — List phases and schemas.
- `POST /reset` — Start a new lifecycle.
- `POST /step` — Submit a biosynthetic action.
- `POST /grader` — Final evaluation.

---

## 🎨 Interactive UI

The **Boardroom Edition** UI provides an immersive view into the organism's life. Monitor the vitality bar, watch failing tests in the live terminal, and inject actions manually to test recovery strategies.

---

## 📄 License

Part of the Meta PyTorch OpenEnv Hackathon submission. Built with ❤️ for the future of self-improving AI.
