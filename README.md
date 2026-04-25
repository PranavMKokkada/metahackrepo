# Autonomous SRE Control Center (OpenEnv)

Autonomous SRE is an OpenEnv-style environment where an agent must keep a self-corrupting service codebase alive under continuous faults.

## Why This Environment

- Long-horizon incident response is still hard for LLM agents.
- This environment turns incident remediation into a step-based control problem with explicit reward feedback.
- It is designed for the OpenEnv hackathon "Self-Improvement" theme.

## What the Agent Sees

- Vitality/SLA proxy score (`0-100`)
- Generated codebase files and checksums
- Unit test status and failing traces
- Checkpoints, watchdog flags, and recent signals
- Dependency graph and incident alerts

## What the Agent Can Do

- `patch_file`
- `run_tests`
- `rollback`
- `quarantine`
- `spawn_subagent`
- `request_expert`
- `emit_signal`
- `do_nothing`

## Reward Design (Current)

- R1: vitality delta
- R2: test recovery
- R3: efficiency bonus
- R4: coordination bonus
- R5: novelty/generalization bonus
- Watchdog penalty for restricted behavior

## Current Measured Results (Policy Evaluation)

These are real rollouts from `results/eval_summary.json` (18 episodes per policy, deterministic seed schedule, held-out seeds used in phase 3).

| Policy | Survival Rate | Mean Reward | Mean Final Vitality | Phase 3 Survival |
|---|---:|---:|---:|---:|
| noop | 0.9444 | -3.9211 | 80.0928 | 0.8333 |
| random | 0.6667 | -1.5042 | 41.5206 | 0.0000 |
| heuristic | 0.6111 | -2.4337 | 33.9483 | 0.0000 |
| stabilized | 0.7222 | -2.6450 | 66.7933 | 0.1667 |
| sft | 0.6667 | -1.3865 | 59.0933 | 0.0000 |

Plot artifacts:

- `results/reward_curve.png`
- `results/baseline_vs_agent.png`
- `results/survival_by_phase.png`
- `results/action_distribution.png`

Important note:

- This repository now includes reproducible evaluation evidence.
- It does **not** claim completed GRPO fine-tuning results yet.

Notebook training log extraction (from a local copy of `Untitled20.ipynb`; not stored in git — use `training/extract_notebook_training.py --notebook path/to/your.ipynb`):

- Logged steps: `60`
- Initial loss: `2.569`
- Final loss: `0.07755`
- Best loss: `0.07705`
- Artifacts: `results/notebook_training_metrics.json`, `results/notebook_training_curve.png`

## Reproduce Results

```bash
python training/train_sft.py --episodes-per-phase 8
python training/evaluate_policy.py --policies noop random heuristic stabilized sft --episodes-per-phase 6 --out-dir results
python training/plot_results.py --results-dir results --summary results/eval_summary.json
python evaluation/run_eval.py
python training/grpo_train.py --mode grpo
```

## Run Locally

```bash
pip install -r requirements.txt
set CODEORGANISM_API_KEYS=change-me
python app.py
```

UI:

- `http://localhost:7860/ui`

## Validate API Loop

```bash
python validate.py --api-url http://127.0.0.1:7860 --api-key change-me
```

Latest local validator run:

- `92/92` checks passed.

## Hugging Face Space

- Configured URL in manifest: `https://huggingface.co/spaces/teletubbies/autonomous-sre`
- Remote public endpoint verification should be run just before submission cutoff.
- GitHub Actions auto-deploy is available via `.github/workflows/deploy-hf-space.yml` (only files required by the root `Dockerfile` are pushed to the Space repo, so large docs and binaries in GitHub never enter Hugging Face git).
- GitHub Actions artifact sync is available via `.github/workflows/upload-hf-artifacts.yml`.
- Required GitHub repository configuration:
  - Secret: `HF_TOKEN` (Hugging Face write token with Space access)
  - Variable: `HF_SPACE_REPO` (example: `teletubbies/autonomous-sre`)
  - Optional variable: `HF_SPACE_BRANCH` (defaults to `main`)
  - Variable: `HF_MODEL_REPO` (example: `teletubbies/autonomous-sre-lora`)
  - Variable: `HF_DATASET_REPO` (example: `teletubbies/autonomous-sre-logs`)
- If CI still shows old vulnerable package versions, you likely used **Re-run job** on an older workflow run (GitHub reuses that run’s commit). Open the latest commit on `main` and use **Actions → Re-run workflow** from there, or push a new commit.

## Demo Asset

- Slide deck/script for 90-120s judge demo: `DEMO_PITCH_SLIDES.md`

## Repository Pointers

- API server: `app.py`
- Environment core: `environment.py`
- Simulator/fault engine: `data.py`
- Reward rubrics: `rubrics.py`
- Evaluation scripts: `training/evaluate_policy.py`, `training/plot_results.py`, `training/rollout.py`, `evaluation/run_eval.py`
- SFT-style policy training: `training/train_sft.py`, output `training/policies/sft_policy.json`
- Training evidence extraction from notebook logs: `training/extract_notebook_training.py`
- GPU runbook and HF Jobs recipe: `training/GRPO_GPU_RUNBOOK.md`, `training/hf_jobs/grpo_job.yaml`
- Results package: `results/`
