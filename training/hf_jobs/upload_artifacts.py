"""Upload training artifacts from an HF Job workspace."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ["HF_TOKEN"]
    model_repo = os.environ["MODEL_REPO"]
    dataset_repo = os.environ["DATASET_REPO"]

    api = HfApi(token=token)

    adapter_dir = Path("results/grpo_run_accuracy/lora_adapter")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(adapter_dir),
        path_in_repo="lora_adapter",
        commit_message="Upload trained LoRA adapter from HF Job",
    )

    metric_pairs = [
        ("training/grpo_gpu_run_result.json", Path("results/grpo_gpu_run_result.json")),
        ("training/training_run_summary.json", Path("results/training_run_summary.json")),
    ]
    for remote_path, local_path in metric_pairs:
        if local_path.exists():
            api.upload_file(
                repo_id=dataset_repo,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                commit_message="Upload training metrics from HF Job",
            )

    print("Uploaded adapter and training metrics.")


if __name__ == "__main__":
    main()
