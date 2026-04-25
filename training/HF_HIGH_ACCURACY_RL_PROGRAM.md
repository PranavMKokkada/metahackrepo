# High-accuracy training program (HF credits) + RL story + deploy

This is the **committed** companion to local `PENDING_WORK_RL_AND_TRAINING_STATUS.md`. It is the operational runbook for maximizing metric quality under **~USD 30 Hugging Face** pay-as-you-go style usage (HF Jobs / Space GPU add-ons).

---

## 1) Exact models to use

### Primary (recommended for LoRA SFT on your JSONL)

| Role | Model ID | Why |
|------|-----------|-----|
| **Train base (4-bit)** | `unsloth/llama-3-8b-Instruct-bnb-4bit` | Already wired in `training/grpo_train.py`; Unsloth + 4-bit fits **T4** VRAM; strong instruction following for patch/run_tests style actions. |
| **Alternative (slightly larger / multilingual)** | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Good if you want stronger reasoning; verify Unsloth compatibility in your TRL/torch stack before switching. |
| **Inference after train (API / router)** | Same base ID **or** merged adapter served via HF Inference / vLLM | `inference.py` defaults `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct` for **router** demos—**after** you train Llama-3 LoRA, point inference to **your** endpoint or merge LoRA for local eval. |

You do **not** need a separate “reward model” checkpoint until you add **true RL** (Section 5). For SFT accuracy, the **environment reward is the evaluation metric**, not the training loss.

---

## 2) Data prerequisites (do this first, every clone)

```bash
python training/generate_sft_data.py
```

Confirm `training/sft_data.jsonl` exists and row count is **high enough** (aim **≥ 2k** lines for stable behavior; regenerate with higher episode counts if the script supports it).

**Quality levers:**

- Cover **all action types** with balanced examples (patch, run_tests, rollback, quarantine, spawn_subagent, emit_signal, do_nothing) — mirrors rubric anti-gaming.
- Include **multi-turn** snippets (fault → wrong action → recovery) not only single-step patches.

---

## 3) “Very high accuracy” training recipe (SFT / LoRA)

**Goal:** lower loss variance, better OOD generalization, stable adapter—**not** fastest run.

### Recommended hyperparameters (HF Jobs / Colab)

| Hyperparameter | Default in repo | **Accuracy-oriented** | Rationale |
|----------------|-----------------|------------------------|-----------|
| `max_steps` | 120 | **400–800** | More optimizer steps on small JSONL. |
| `learning_rate` | 2e-4 | **8e-5 to 1.5e-4** | Reduces catastrophic forgetting / overshoot. |
| `lora_r` / `lora_alpha` | 16 / 16 | **32 / 32** (or 64/64 if VRAM allows) | Higher rank = more expressivity for tool-use formatting. |
| `per_device_train_batch_size` | 2 | **1** if OOM | Stability > speed. |
| `gradient_accumulation_steps` | 4 | **8–16** | Effective larger batch without VRAM spike. |
| `warmup_steps` | 10 | **20–50** | Safer early updates. |
| `max_seq_length` | 2048 | **2048–4096** | Only raise if examples need it (VRAM cost). |
| `bf16` | off (fp16) | **`--bf16`** on A100/L4/H100 | Prefer bf16 on capable GPUs. |
| `save_steps` | 40 | **every 50–100 steps** | Keep best checkpoint for offline eval. |

### One-shot command (Linux GPU job)

```bash
pip install -r requirements-training.txt

python training/grpo_train.py \
  --mode grpo \
  --run-gpu \
  --bf16 \
  --model-id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --dataset-path training/sft_data.jsonl \
  --output-dir results/grpo_run_accuracy \
  --max-seq-length 2048 \
  --max-steps 600 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 12 \
  --warmup-steps 40 \
  --logging-steps 10 \
  --save-steps 100 \
  --lora-r 32 \
  --lora-alpha 32 \
  --seed 42
```

Artifacts:

- `results/grpo_run_accuracy/lora_adapter/` — **push this folder** to your HF model repo (see Section 6).
- `results/grpo_gpu_run_result.json` — training metrics for the README / deck.

### HF Jobs (spend ~$30 wisely)

1. Use **`training/hf_jobs/grpo_job_accuracy.yaml`** (this repo) as the template—it uses **longer steps** and **accuracy-oriented** flags.
2. Pick **hardware** vs budget:
   - **`t4-small`** — cheapest; use **long wall time** + smaller batch (script above).
   - **`t4-medium` / `l4x1`** — better step/sec if credits allow fewer total hours.
3. From repo root (with `HF_TOKEN`):

```bash
hf jobs run --from-repo . --config training/hf_jobs/grpo_job_accuracy.yaml
```

(Adjust to your actual `hf jobs` CLI version; older flows used raw `--command` strings—see `training/GRPO_GPU_RUNBOOK.md`.)

---

## 4) Where to run (GPU matrix)

| Platform | When to use |
|----------|-------------|
| **HF Jobs** | Best match for “**~$30 credits**”, reproducible logs, no Colab disconnects. |
| **Google Colab Pro+** | If you already pay Google; same commands after `pip install -r requirements-training.txt`. |
| **Local Windows** | Use **`requirements-training-win311.txt`** for CPU sanity only; **do not** expect Unsloth/xformers parity—train on Linux GPU. |

---

## 5) RL: how it fits today vs the “wow” upgrade

### Today (shipped)

- **Environment = implicit MDP** with **dense + sparse reward engineering** (`rubrics.py`).
- **Learning = offline SFT** on structured trajectories (`training/sft_data.jsonl`) → **LoRA adapter**.
- **Evaluation = Monte Carlo rollouts** (`training/rollout.py`) → metrics that map directly to hackathon “**improvement in rewards**”.

### Wow-factor RL (next engineering milestone)

1. **Sample** trajectories: policy π produces actions in `CodeOrganismEnv`; log **(s, a, r, s')** at scale.
2. **Optimize** π with **GRPO / PPO / RLOO** (TRL) using **episode return** or **shaped step reward** as the scalar signal.
3. **Show** a learning curve: mean return vs training iteration vs **fixed** baseline seeds.

That is the **honest RL story**: same environment, same rubric, but **policy gradients** instead of only cross-entropy on static JSONL.

---

## 6) After training: where to deploy the model

| Destination | What to upload | How |
|---------------|----------------|-----|
| **HF Model repo** (e.g. `YOUR_USER/autonomous-sre-lora`) | Contents of `results/grpo_run_accuracy/lora_adapter/` + a small `README.md` describing base model ID | `huggingface_hub` `upload_folder` or `hf upload`; GitHub workflow `upload-hf-artifacts.yml` already syncs `training/policies/`—extend similarly for `lora_adapter/` if desired. |
| **HF Inference Endpoints** (optional) | Merged or adapter-backed server | For live judge demos beyond Space CPU limits. |
| **This repo’s Space** | Does **not** embed giant weights; Space runs **API + UI**. Keep adapter on **Model** repo; Space or `inference.py` loads via **HF token** + model id. |

---

## 7) Post-train verification (required for “high accuracy” claims)

```bash
python training/evaluate_policy.py \
  --policies noop random heuristic stabilized sft \
  --episodes-per-phase 12 \
  --out-dir results

python training/plot_results.py --results-dir results --summary results/eval_summary.json
python evaluation/run_eval.py
```

Until the **SFT policy** is driven by the **new LoRA** (requires wiring `rollout.py` / API to HF inference or merged weights), these numbers still measure **table SFT** + baselines. Plan a **`lora`** policy mode once inference is connected.

---

## 8) Notebook extraction (optional)

If `Untitled20.ipynb` is only local, either:

- pass `--notebook` when you add a thin wrapper, or  
- rely on GPU run metrics in `grpo_gpu_run_result.json` as the primary training evidence.

---

*Keep this file in git so teammates and judges can follow the same high-accuracy path.*
