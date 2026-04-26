# GRPO GPU Runbook (Colab + HF Jobs)

This runbook turns `training/grpo_train.py` into a reproducible GPU training workflow.

## 1) Prerequisites

- GPU runtime (Colab T4/A100 or HF Jobs T4+)
- Python 3.12 (recommended for current GPU runtimes; 3.11 also works for backend-serving workflows)
- Repo checkout with `training/sft_data.jsonl`

## 2) Install Dependencies

```bash
pip install -r requirements-training.txt
```

For Windows local CPU sanity checks, use:

```bash
pip install -r requirements-training-win312.txt
```

## 3) Recipe Mode (no training, generates config artifacts)

```bash
python training/grpo_train.py --mode grpo
```

Outputs:

- `results/grpo_gpu_recipe.json`
- `results/training_run_summary.json`
- `results/notebook_training_metrics.json`
- `results/notebook_training_curve.png`

## 4) Actual GPU Training Run

```bash
python training/grpo_train.py \
  --mode grpo \
  --run-gpu \
  --model-id unsloth/llama-3-8b-Instruct-bnb-4bit \
  --dataset-path training/sft_data.jsonl \
  --output-dir results/grpo_run \
  --max-seq-length 2048 \
  --max-steps 120 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 10 \
  --logging-steps 5 \
  --save-steps 40 \
  --lora-r 16 \
  --lora-alpha 16
```

Additional output:

- `results/grpo_gpu_run_result.json`
- `results/grpo_run/lora_adapter/` (adapter checkpoint)

## 5) HF Jobs Command Template

From `results/grpo_gpu_recipe.json`:

```bash
hf jobs run --flavor t4-small --env MODEL_ID=unsloth/llama-3-8b-Instruct-bnb-4bit --command "python training/grpo_train.py --run-gpu --dataset-path training/sft_data.jsonl --output-dir results/grpo_run"
```

## 6) Post-Training Evaluation

```bash
python training/evaluate_policy.py --policies noop random heuristic stabilized sft --episodes-per-phase 6 --out-dir results
python training/plot_results.py --results-dir results --summary results/eval_summary.json
python evaluation/run_eval.py
```

## Notes

- Current implementation uses SFT-style LoRA fine-tuning in the GPU path as a practical baseline.
- Keep GRPO nomenclature in roadmap as preferred path; this script now provides an executable GPU recipe and training runtime integration.
