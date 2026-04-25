# Autonomous SRE Control Center — Instructional Context

Autonomous SRE Control Center is an industry-grade reinforcement learning (RL) training environment built on **Meta OpenEnv**. It transforms an LLM agent into a Site Reliability Engineer responsible for maintaining **SLA Compliance** inside a hostile, self-corrupting cloud sandbox managed by a **Chaos Engine**.

## 📌 Project Overview

- **Core Goal:** Train LLM agents to autonomously diagnose, remediate, and prevent infrastructure failures.
- **Main Technologies:**
    - **Backend:** Python 3.11+, FastAPI (SRE Engine Server).
    - **RL Framework:** OpenEnv, TRL (GRPO Trainer), Unsloth (QLoRA).
    - **Sandboxing:** Physical staging in `/tmp` with `os.chmod` read-only guardrails.
    - **Expert Oracle:** Real-time patch validation via **OpenAI GPT-4o-mini**.
    - **Frontend:** Startup-grade Gradio Dashboard (SRE Control Center).
- **Architecture:** 
    - A FastAPI server (`app.py`) manages SRE sessions.
    - The engine (`environment.py`) tracks SLA Compliance, MTTR, and Business Impact.
    - A Chaos Simulator (`data.py`) procedurally generates microservice topologies and injects architectural faults.
    - Observation space includes a **World Model** (Dependency Graph) for Root Cause Analysis.

## 🚀 Key Commands

### Environment & UI
- **Run Server:** `python app.py` (Runs at `http://localhost:7860`, UI at `/ui`)
- **Validate Spec:** `python validate.py` (Checks compliance with OpenEnv standard)
- **Run Logic Tests:** `pytest test_env.py` (Verifies remediation protocols and SLA rewards)

### Training Pipeline
- **Generate SFT Data:** `python training/generate_sft_data.py` (Creates synthetic SRE remediation traces)
- **Start RL Training:** `python training/grpo_train.py` (Starts the Chaos-driven GRPO training loop)

## 🛰️ SRE Dynamics

### SLA Compliance & Metrics
The agent starts with 100% SLA Index. Every intervention has a budget cost:
- `patch_node`: −2.0
- `run_diagnostics`: −3.0
- `spawn_team`: −5.0
- `circuit_break`: −1.0
- `rollback_canary`: −4.0
- `oracle_query`: −6.0

**Business Impact:** Every successful remediation saves **300s** of simulated downtime.

### Rewards (SRE Rubric)
1. **SLA Stability (35%)** - Maintaining system health index.
2. **Recovery Success (30%)** - Restoring degraded services (FAIL→PASS).
3. **Remediation Efficiency (15%)** - Solving incidents with minimal interventions.
4. **Teaming Bonus (10%)** - Rewarded (+0.5) for signaling **Intent** before acting.
5. **Architecture Generalization (10%)** - Performance on held-out topological seeds.

## 🛠 Development Conventions

- **World Modeling:** The `dependency_graph` in `data.py` must be updated if new microservice templates are added.
- **Security (Watchdog):** Enforces strict penalties (−10.0) for unauthorized access to system files or test configurations.
- **Explainability:** All remediation protocols require a `justification` string, which is surfaced in the Control Center UI.
- **Chaos Triggers:** New failure vectors should be added to the `FaultInjector` catalog in `data.py` and mapped to the manual UI trigger.
