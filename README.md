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

---

## 🔗 Submission Materials

| Item | Link |
|------|------|
| **Hugging Face Space** | [Deploy Link Here](https://huggingface.co/spaces/Andrea/code-organism-vm) |
| **Demo Video** | [Watch on YouTube](https://youtube.com/...) |
| **Technical Blog** | [Read on Hugging Face](https://huggingface.co/blog/...) |
| **Training Notebook** | [Open in Colab (Local Link)](CodeOrganismVM_Training.ipynb) |

---

## 👨‍⚖️ Judging & Evaluation

### Judging Overview

**Evaluation:** Teams will be scored based on the following criteria:
- **Environment Innovation (40%):** Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?
- **Storytelling (30%):** Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?
- **Showing Improvement in Rewards (20%):** Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)?
- **Reward and Training Script/Pipeline Setup (10%):** Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent’s inference (how it acts in the environment)?

---

## 🏆 OpenEnv Hackathon - What Judges Look For

This guide tells you what makes a strong submission for the **OpenEnv Hackathon (India 2026)**. Read it before you start building, and again before you submit.

> [!NOTE]
> Please remember only one submission per team. If you have multiple ideas, pick the best one and go for it. Please make sure that the **URL link of your environment is submitted** as judges will pull the environment from the URL to evaluate it. Changes or commits after the submission deadline will not be considered.

### 📝 TL;DR
Build an environment that an LLM could actually be trained on to get measurably better at something interesting. Then show that training. Then tell the story.
*A messy but ambitious environment with real training evidence beats a polished but boring one.*

---

### ⚖️ Detailed Judging Criteria

#### 1. Environment Innovation (40%)
*   Is the environment novel, creative, or genuinely challenging?
*   Does it meaningfully test agent behavior in a way that hasn't been done before?

#### 2. Storytelling & Presentation (30%)
*   Can you clearly explain the problem, the environment, and what the agent learned?
*   Is the demo engaging and easy to follow for a non-technical audience?

#### 3. Showing Improvement in Rewards (20%)
*   Is there observable evidence of training progress? 
*   Reward curves, before/after behavior, comparison against a baseline -- anything that proves the agent learned something.

#### 4. Reward & Training Pipeline (10%)
*   Is the reward logic coherent? 
*   Does the pipeline produce meaningful improvement in the trained agent's behavior?

---

### ⚠️ Minimum Submission Requirements
*These are non-negotiable. Submissions missing any of these are at a serious disadvantage.*

1.  **Use OpenEnv (latest release):** Build on top of the framework; don’t reinvent the wheel.
2.  **Training Script:** A working training script using Unsloth or Hugging Face TRL, ideally as a Colab notebook.
3.  **Training Evidence:** At minimum, loss and reward plots from a real run.
4.  **Short Writeup:** A mini-blog on Hugging Face, a < 2 min YouTube video, or a short slide deck.
5.  **Hugging Face Space:** Push your environment to a HF Space so it’s discoverable and runnable.
6.  **Comprehensive README:** Motivate the problem, explain the env, and show results. **Must link to the HF Space and all additional materials.**

---

### 🌟 What Makes a Submission Stand Out

#### 🚀 Pick an Ambitious, Original Problem
The themes (problems) are deliberately open. Don't build chess, snake, or grid-world clones.
*   Does this environment teach an LLM something it currently can’t do well?
*   Is the domain underexplored in RL/LLM training?
*   Could a researcher write a paper about training on this?

#### 🎯 Design a Reward Signal that Teaches
*   Provides a rich, informative signal (not just 0/1 at the end).
*   Captures something hard to measure in a clever way.
*   Uses **OpenEnv’s Rubric system** thoughtfully (composable rubrics > monolithic scoring).
*   Is hard to game (no reward hacking).

#### 📈 Show Real Training, End to End
*   Training loop must connect to your environment (not a static dataset).
*   Train long enough that the curves mean something.
*   Compare trained agent vs. baseline (quantitative and qualitative).
*   Include plots/numbers in README.

#### 📊 Make Your Plots Readable
*   Label both axes and include units.
*   Save plots as `.png` or `.jpg` and commit them to the repo.
*   Embed key plots in README with clear captions.

#### 📖 Tell a Story, Not an API Doc
Your README/blog should answer:
1.  **Problem:** What capability gap are you targeting?
2.  **Environment:** What does the agent see, do, and get rewarded for?
3.  **Results:** What changed after training? Show it.
4.  **Impact:** Who would care, and why?

#### 🛠️ Engineer it Cleanly
*   Use OpenEnv’s `Environment` / `MCPEnvironment` base classes properly.
*   Respect client/server separation.
*   Follow standard Gym-style API (`reset`, `step`, `state`).
*   Have a valid `openenv.yaml` manifest.
*   Avoid reserved tool names.

---

**Final Note:** Judges are looking for environments that push the frontier of what we can train LLMs to do. Be ambitious. Good luck!

