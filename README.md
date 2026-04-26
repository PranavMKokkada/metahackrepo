# Autonomous SRE Control Center (OpenEnv)

Autonomous SRE is an OpenEnv-style environment where an agent must keep a self-corrupting service codebase alive under continuous faults.

## Why This Environment

- Long-horizon incident response is still hard for LLM agents.
- This environment turns incident remediation into a step-based control problem with explicit reward feedback.
- It is designed for the OpenEnv hackathon "Self-Improvement" theme.

## Hackathon Non-Negotiables Coverage

This section maps directly to the OpenEnv India Hackathon 2026 finale requirements.

- OpenEnv usage: implemented via OpenEnv-style API + `openenv.yaml`.
- Working training path (Unsloth/TRL): `training/grpo_train.py` and `CodeOrganismVM_Training.ipynb`.
- Real training evidence: imported LoRA run artifacts (`grpo_gpu_run_result.json`, logs, environment dump) + evaluation plots.
- Discoverable runnable environment: Hugging Face Space at `https://huggingface.co/spaces/teletubbies/autonomous-sre`.
- README with motivation + env behavior + evidence: this file.
- Additional materials linked from README: pitch deck (`DEMO_PITCH_SLIDES.md`) and notebook (`CodeOrganismVM_Training.ipynb`).

Final pre-submission gate:

```bash
python scripts/submission_preflight.py
```

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
- LoRA fine-tuning evidence can be imported from a completed external GPU run bundle.

Notebook training log extraction (from local notebook runs):

- Logged steps: `60`
- Initial loss: `2.569`
- Final loss: `0.07755`
- Best loss: `0.07705`
- Artifacts: `results/notebook_training_metrics.json`, `results/notebook_training_curve.png`
- Notebook assets: `CodeOrganismVM_Training.ipynb` (project notebook), `Untitled20.ipynb` (local run log notebook)

## Reproduce Results

```bash
python training/train_sft.py --episodes-per-phase 8
python training/evaluate_policy.py --policies noop random heuristic stabilized sft --episodes-per-phase 6 --out-dir results
python training/plot_results.py --results-dir results --summary results/eval_summary.json
python evaluation/run_eval.py
python training/grpo_train.py --mode grpo
```

Notebook re-run option:

```bash
jupyter notebook CodeOrganismVM_Training.ipynb
```

## Python Version Split (Recommended)

- Training pipeline: Python `3.12` (GPU environments like Lightning/HF Jobs).
- Backend/API serving: Python `3.11` or `3.12` with `requirements.txt`.
- Keep training and backend in separate virtual environments and only move model artifacts (`lora_adapter`, tokenizer files, evaluation JSON/plots) between them.

## Import Completed LoRA Bundle

If you already finished GPU training externally, import the downloadable bundle:

```bash
python scripts/import_lora_bundle.py --bundle "CodeOrganismVM_lora_training_bundle (1).tar.gz" --copy-dataset
```

This wires:

- `artifacts/runtime/lora_adapter/` (local stable adapter path)
- `artifacts/lora_bundle/results/*.json|*.log` (training evidence)
- `training/sft_data.jsonl` (optional copy from the bundle with `--copy-dataset`)

Quick local hardware readiness check for base+LoRA inference:

```bash
python scripts/check_local_lora_compute.py
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

### Publish imported LoRA artifacts to Hugging Face repos

Set environment variables and upload adapter + run logs:

```bash
# Linux/macOS
export HF_TOKEN=...
export HF_MODEL_REPO=teletubbies/autonomous-sre-lora
export HF_DATASET_REPO=teletubbies/autonomous-sre-logs
export BASE_MODEL_ID=unsloth/llama-3-8b-Instruct-bnb-4bit
python scripts/publish_lora_artifacts.py
```

```powershell
# Windows PowerShell
$env:HF_TOKEN="..."
$env:HF_MODEL_REPO="teletubbies/autonomous-sre-lora"
$env:HF_DATASET_REPO="teletubbies/autonomous-sre-logs"
$env:BASE_MODEL_ID="unsloth/llama-3-8b-Instruct-bnb-4bit"
python scripts/publish_lora_artifacts.py
```

## Demo Asset

- Slide deck/script for 90-120s judge demo: `DEMO_PITCH_SLIDES.md`
- Guided recording mode in UI: `Run Guided Demo Mode` button

## External Submission Links (fill before final form submit)

- Hugging Face Space URL: `https://huggingface.co/spaces/teletubbies/autonomous-sre`
- Model repo URL: `https://huggingface.co/teletubbies/autonomous-sre-lora`
- Dataset repo URL: `https://huggingface.co/datasets/teletubbies/autonomous-sre-logs`
- Optional HF blog post URL: `ADD_YOUR_HF_BLOG_URL`
- Optional <2 min YouTube demo URL: `ADD_YOUR_YOUTUBE_URL`

Important:

- Do not upload large video binaries to the environment repo/Space.
- Use public URLs for blog/video references instead.

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
