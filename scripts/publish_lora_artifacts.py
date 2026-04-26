#!/usr/bin/env python3
"""Upload imported LoRA adapter + run artifacts to Hugging Face repos."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    hf_token = _required_env("HF_TOKEN")
    model_repo = _required_env("HF_MODEL_REPO")
    dataset_repo = _required_env("HF_DATASET_REPO")
    base_model_id = os.getenv("BASE_MODEL_ID", "unsloth/llama-3-8b-Instruct-bnb-4bit")

    adapter_dir = Path("artifacts/runtime/lora_adapter")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            "Adapter directory not found. Run scripts/import_lora_bundle.py first."
        )

    api = HfApi(token=hf_token)

    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(adapter_dir),
        path_in_repo="lora_adapter",
        commit_message="Upload LoRA adapter from local bundle import",
    )

    readme = Path("artifacts/runtime/MODEL_CARD.md")
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "\n".join(
            [
                "# Autonomous SRE LoRA Adapter",
                "",
                f"- Base model: `{base_model_id}`",
                "- Adapter path in this repo: `lora_adapter/`",
                "- Format: PEFT LoRA (not merged full checkpoint)",
                "",
                "Use with `peft.PeftModel.from_pretrained(base_model, adapter_path)`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    api.upload_file(
        repo_id=model_repo,
        repo_type="model",
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        commit_message="Add base-model usage note",
    )

    artifact_files = [
        ("training/grpo_gpu_run_result.json", Path("artifacts/lora_bundle/results/grpo_gpu_run_result.json")),
        ("training/grpo_gpu_recipe.json", Path("artifacts/lora_bundle/results/grpo_gpu_recipe.json")),
        ("training/full_training.log", Path("artifacts/lora_bundle/results/full_training.log")),
        ("training/training_environment.json", Path("artifacts/lora_bundle/results/training_environment.json")),
        (
            "training/training_environment_pip_freeze.txt",
            Path("artifacts/lora_bundle/results/training_environment_pip_freeze.txt"),
        ),
        ("training/sft_data.jsonl", Path("artifacts/lora_bundle/training/sft_data.jsonl")),
    ]
    for remote_path, local_path in artifact_files:
        if local_path.exists():
            api.upload_file(
                repo_id=dataset_repo,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                commit_message="Upload LoRA run artifacts from local bundle import",
            )

    print(f"Uploaded adapter to model repo: {model_repo}")
    print(f"Uploaded logs/metrics to dataset repo: {dataset_repo}")


if __name__ == "__main__":
    main()
