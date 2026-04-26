# Training an autonomous SRE agent on CodeOrganismVM

**OpenEnv / self-improvement angle:** incident response is a long-horizon control problem. We treat a **self-corrupting service codebase** as the world, expose it through a standard **step / reset / observation** API, and train policies that maximize explicit **SRE rubrics** (vitality, test recovery, efficiency, coordination) instead of proxy loss alone.

---

## What we built

**CodeOrganismVM** is a sandbox where an agent (LLM or classical policy) must keep a synthetic codebase alive under injected faults: flaky tests, corrupted symbols, dependency pressure, and watchdog rules. The environment implements:

- **Vitality** as a running SLA-style health index  
- **Deterministic fault injection** across curriculum phases (`phase_1` → `phase_3`)  
- **Actions** such as `patch_file`, `run_tests`, `rollback`, `quarantine`, `spawn_subagent`, `request_expert`, and `emit_signal`  
- **Reward decomposition** aligned with rubrics R1–R5 so training signal matches what judges care about  

The **live demo** runs on **Hugging Face Spaces** (Docker): same API, a **Gradio control center** for operators, and a **static console** for production-mode and guardrail workflows.

---

## Training stack

We document and run training from a dedicated notebook that wires **Unsloth** + **TRL** for efficient fine-tuning, with **GRPO-style** group optimization over rollouts against the real environment step function (not a hand-written toy MDP).

- **Notebook (canonical training entrypoint):** `CodeOrganismVM_Training.ipynb` in this repository  
- **Pinned training dependencies:** `requirements-training.txt` (Torch, TRL, PEFT, etc.)  
- **Offline / GPU job path:** `training/grpo_train.py`, `training/GRPO_GPU_RUNBOOK.md`, and optional Hugging Face Jobs YAML under `training/hf_jobs/`  

The notebook stresses running from a **checked-out repo root** so cells import the same `environment.py`, `models.py`, and data the Space ships—no shadow fork of the env.

---

## Evidence we ship

- **Evaluation harness:** `training/evaluate_policy.py` + `results/eval_summary.json` for survival and reward by phase  
- **Plots:** reward curves, survival by phase, action distributions under `results/`  
- **Optional LoRA bundle import:** `scripts/import_lora_bundle.py` to attach external GPU run artifacts without bloating the Space Docker context  

---

## Why Hugging Face

- **Spaces** give a **public, incognito-friendly** URL for judges to drive the environment  
- **Models / datasets** repos (separate from the Space) store adapters and logs without coupling them to the runtime image  
- **Git-backed** Space repos let us publish this **markdown blog** and the **training notebook** as first-class files—no paywall, no login wall for read-only browsing  

---

## Links (replace `ORG/SPACE` with your `HF_SPACE_REPO`, e.g. `teletubbies/autonomous-sre`)

| Asset | URL pattern |
|--------|-------------|
| **Running Space (demo)** | `https://huggingface.co/spaces/ORG/SPACE` |
| **Training notebook (file)** | `https://huggingface.co/spaces/ORG/SPACE/blob/main/CodeOrganismVM_Training.ipynb` |
| **This blog post (file)** | `https://huggingface.co/spaces/ORG/SPACE/blob/main/BLOG_POST.md` |
| **Source of truth (GitHub)** | Your public repository URL on GitHub |

After each deploy from CI, open the **blob** links in an **Incognito** window to confirm they load without authentication.

---

## Closing

We are not “vibe-training” a chatbot—we are **aligning a policy to measurable remediation outcomes** in a hostile, stateful simulator. That is the story this notebook and the Space are meant to tell together.
