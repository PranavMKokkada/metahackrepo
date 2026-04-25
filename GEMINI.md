# CodeOrganismVM — Instructional Context

CodeOrganismVM is a reinforcement learning (RL) training environment where an LLM agent (the "organism") operates inside a continuously corrupting codebase (the "host"). The agent must self-heal, self-correct, and thrive to maintain its **Vitality Score [0–100]** while battling a **FaultInjector** adversary.

## 📌 Project Overview

- **Core Goal:** Train LLM agents to survive and maintain system health in hostile environments.
- **Main Technologies:**
    - **Backend:** Python 3.11+, FastAPI (Environment Server).
    - **RL Framework:** OpenEnv, TRL (GRPO Trainer), Unsloth (QLoRA).
    - **Sandboxing:** Docker (Isolated execution), OverlayFS (Protected file zones).
    - **Frontend:** Gradio (Boardroom Edition UI).
- **Architecture:** 
    - A FastAPI server (`app.py`) manages environment sessions.
    - The core logic (`environment.py`) tracks vitality, costs, and fault injection intervals.
    - A simulator (`data.py`) procedurally generates broken codebases and injects faults.
    - Training scripts (`training/`) handle SFT data generation and RL optimization via GRPO.

## 🚀 Key Commands

### Environment & UI
- **Run Server:** `python app.py` (Runs at `http://localhost:7860`, UI at `/ui`)
- **Validate Spec:** `python validate.py` (Checks compliance with the hackathon spec)
- **Run Tests:** `pytest test_env.py` (Verifies environment dynamics and rewards)

### Training Pipeline
- **Generate SFT Data:** `python training/generate_sft_data.py` (Creates synthetic expert traces)
- **Start RL Training:** `python training/grpo_train.py` (Starts the GRPO training loop)

## 🧬 Environment Dynamics

### Vitality & Costs
The agent starts with 100 Vitality. Every action has a cost:
- `patch_file`: −2.0
- `run_tests`: −3.0
- `spawn_subagent`: −5.0
- `quarantine`: −1.0
- `rollback`: −4.0
- `request_expert`: −6.0

### Rewards (R1–R5)
The system optimizes for five dimensions:
1. **R1: Vitality (35%)** - Health maintenance.
2. **R2: Recovery (30%)** - Fixing failing tests.
3. **R3: Efficiency (15%)** - Minimal actions via $1/\sqrt{n}$.
4. **R4: Coordination (10%)** - Effective subagent delegation.
5. **R5: Generalization (10%)** - Performance on held-out seeds.

### Phases
- **Phase 1:** Single fault every 8 steps.
- **Phase 2:** Multi-fault (2–4) every 6 steps.
- **Phase 3:** Adversarial (adaptive) every 4 steps.

## 🛠 Development Conventions

- **Models:** All data structures are defined in `models.py` using Pydantic.
- **Security:** The `Watchdog` layer in `environment.py` penalizes attempts to modify protected files (like tests) or escape the sandbox.
- **Action Format:** Agents must emit actions in a structured JSON format compatible with the `Action` schema.
- **Atomic Operations:** State mutations (patching, rollback) use transaction locks to ensure consistency against the background FaultInjector.
- **Adding Faults:** New fault types should be added to the `FaultInjector` catalog in `data.py`.
