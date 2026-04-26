#!/usr/bin/env bash
set -euo pipefail

# Lightning A100 training runner (Python 3.12)
# - Sets up a local venv
# - Logs into Hugging Face (token from arg or HF_TOKEN env var)
# - Installs training deps
# - Generates SFT data
# - Runs accuracy-oriented GPU training
# - Runs post-training evaluation
#
# Usage examples:
#   bash training/lightning_a100_run.sh --hf-token "$HF_TOKEN"
#   HF_TOKEN=hf_xxx bash training/lightning_a100_run.sh
#   bash training/lightning_a100_run.sh --skip-eval
#
# Note:
#   This script never writes the token into git-tracked files.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_TOKEN_INPUT="${HF_TOKEN:-}"
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      HF_TOKEN_INPUT="${2:-}"
      shift 2
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash training/lightning_a100_run.sh [--hf-token <token>] [--skip-eval]"
      exit 1
      ;;
  esac
done

if [[ -z "${HF_TOKEN_INPUT}" ]]; then
  echo "ERROR: Hugging Face token is required."
  echo "Pass via --hf-token <token> or export HF_TOKEN=<token>."
  exit 1
fi

echo "==> Repo root: $REPO_ROOT"
echo "==> Creating Python 3.12 venv (.venv312) if missing..."
if [[ ! -x ".venv312/bin/python" ]]; then
  python3.12 -m venv .venv312
fi

PYTHON_BIN=".venv312/bin/python"

echo "==> Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

echo "==> Installing training dependencies..."
"$PYTHON_BIN" -m pip install -r requirements-training.txt

echo "==> Authenticating Hugging Face..."
HF_TOKEN_INPUT="$HF_TOKEN_INPUT" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import login, whoami

token = os.environ["HF_TOKEN_INPUT"]
login(token=token, add_to_git_credential=True)
info = whoami()
name = info.get("name") or info.get("fullname") or "unknown-user"
print(f"Authenticated as: {name}")
PY

echo "==> Generating SFT dataset..."
"$PYTHON_BIN" training/generate_sft_data.py

echo "==> Starting accuracy-oriented A100 training..."
"$PYTHON_BIN" training/grpo_train.py \
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

if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "==> Running evaluation and plots..."
  "$PYTHON_BIN" training/evaluate_policy.py --policies noop random heuristic stabilized sft --episodes-per-phase 12 --out-dir results
  "$PYTHON_BIN" training/plot_results.py --results-dir results --summary results/eval_summary.json
  "$PYTHON_BIN" evaluation/run_eval.py
fi

echo ""
echo "Done. Key outputs:"
echo "  - results/grpo_run_accuracy/lora_adapter/"
echo "  - results/grpo_gpu_run_result.json"
echo "  - results/eval_summary.json (if eval ran)"
echo "  - results/*.png (if eval ran)"
echo "  - evaluation/report.md (if eval ran)"
