"""GPU-ready GRPO/SFT training entrypoint for CodeOrganismVM.

This script provides:
1) a reproducible GPU training recipe for Colab / HF Jobs
2) optional execution of LoRA fine-tuning when Unsloth/TRL deps are present
3) notebook log extraction for evidence artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List


def _run_notebook_metrics_extraction() -> dict:
    cmd = [
        sys.executable,
        "training/extract_notebook_training.py",
        "--notebook",
        "Untitled20.ipynb",
        "--out-json",
        "results/notebook_training_metrics.json",
        "--out-plot",
        "results/notebook_training_curve.png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

def _write_training_summary(mode: str, model_id: str, notebook_extract: dict) -> str:
    os.makedirs("results", exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_id": model_id,
        "notebook_extract": notebook_extract,
        "notes": (
            "This run materializes notebook-derived training evidence. "
            "Use external GPU runtime for full GRPO fine-tuning."
        ),
    }
    out_path = os.path.join("results", "training_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _load_sft_records(dataset_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _run_gpu_sft(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except Exception as exc:  # pragma: no cover - dependency gated
        return {
            "ran": False,
            "reason": f"Missing GPU training dependencies: {exc}",
            "hint": "Install requirements-training.txt in a GPU runtime.",
        }

    records = _load_sft_records(args.dataset_path)
    texts = [{"text": _messages_to_text(r.get("messages", []))} for r in records]
    dataset = Dataset.from_list(texts)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            bf16=args.bf16,
            fp16=not args.bf16,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            save_strategy="steps",
            save_steps=args.save_steps,
        ),
    )
    train_result = trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    metrics = getattr(train_result, "metrics", {})
    return {
        "ran": True,
        "records_used": len(records),
        "output_dir": args.output_dir,
        "metrics": metrics,
    }


def _write_gpu_recipe(args: argparse.Namespace) -> str:
    recipe = {
        "model_id": args.model_id,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "bf16": args.bf16,
        "seed": args.seed,
        "colab_command": (
            "pip install -r requirements-training.txt && "
            f"python training/grpo_train.py --run-gpu --model-id {args.model_id} "
            f"--dataset-path {args.dataset_path} --output-dir {args.output_dir}"
        ),
        "hf_jobs_command": (
            "hf jobs run --flavor t4-small "
            f"--env MODEL_ID={args.model_id} "
            f"--command \"python training/grpo_train.py --run-gpu --dataset-path {args.dataset_path} --output-dir {args.output_dir}\""
        ),
    }
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "grpo_gpu_recipe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grpo", "sft"], default="grpo")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "unsloth/llama-3-8b-Instruct-bnb-4bit"))
    parser.add_argument("--dataset-path", default="training/sft_data.jsonl")
    parser.add_argument("--output-dir", default="results/grpo_run")
    parser.add_argument("--run-gpu", action="store_true", help="Run actual GPU fine-tuning if deps are available.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Initializing {args.mode.upper()} training for {args.model_id}...")
    notebook_extract = _run_notebook_metrics_extraction()
    if notebook_extract["returncode"] == 0:
        print("Notebook training metrics extracted to results/")
    else:
        print("Notebook extraction failed; see summary for stderr.")
    recipe_path = _write_gpu_recipe(args)
    print(f"Wrote GPU recipe: {recipe_path}")
    summary_path = _write_training_summary(args.mode, args.model_id, notebook_extract)
    print(f"Wrote training summary: {summary_path}")
    if args.run_gpu:
        run_info = _run_gpu_sft(args)
        os.makedirs("results", exist_ok=True)
        run_info_path = os.path.join("results", "grpo_gpu_run_result.json")
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        print(f"Wrote GPU run result: {run_info_path}")
        if not run_info.get("ran"):
            print("GPU training did not run:", run_info.get("reason"))
    else:
        print("Recipe mode only. Use --run-gpu in Colab/HF Jobs runtime.")


if __name__ == "__main__":
    main()
