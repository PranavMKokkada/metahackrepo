#!/usr/bin/env python3
"""Sanity checks for end-to-end project readiness after LoRA import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _assert_exists(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {p}")


def main() -> None:
    required_paths = [
        "artifacts/runtime/lora_adapter/adapter_model.safetensors",
        "artifacts/runtime/lora_adapter/adapter_config.json",
        "artifacts/lora_bundle/results/grpo_gpu_run_result.json",
        "artifacts/lora_bundle/results/training_environment.json",
        "training/sft_data.jsonl",
    ]
    for p in required_paths:
        _assert_exists(p)

    with open("artifacts/lora_bundle/results/grpo_gpu_run_result.json", "r", encoding="utf-8") as f:
        run_result = json.load(f)
    if not run_result.get("ran"):
        raise RuntimeError("Imported training run reports ran=false.")

    test_python = Path(sys.executable)
    try:
        __import__("pytest")
    except Exception:
        fallback = Path(".venv311/Scripts/python.exe")
        if fallback.exists():
            test_python = fallback
        else:
            raise RuntimeError(
                "pytest is not available in the current interpreter and .venv311 was not found."
            )

    test_cmd = [str(test_python), "-m", "pytest", "-q"]
    result = subprocess.run(test_cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("pytest failed.")

    print(json.dumps({
        "ok": True,
        "records_used": run_result.get("records_used"),
        "train_loss": (run_result.get("metrics") or {}).get("train_loss"),
    }, indent=2))


if __name__ == "__main__":
    main()
