#!/usr/bin/env python3
"""Import a downloaded LoRA bundle into local runtime paths."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and wire LoRA training bundle artifacts.")
    parser.add_argument(
        "--bundle",
        default="CodeOrganismVM_lora_training_bundle (1).tar.gz",
        help="Path to downloaded bundle tar.gz",
    )
    parser.add_argument(
        "--extract-dir",
        default="artifacts/lora_bundle",
        help="Where to extract the bundle",
    )
    parser.add_argument(
        "--runtime-adapter-dir",
        default="artifacts/runtime/lora_adapter",
        help="Stable adapter directory used by local runtime scripts",
    )
    parser.add_argument(
        "--copy-dataset",
        action="store_true",
        help="Copy training/sft_data.jsonl from bundle into training/sft_data.jsonl",
    )
    args = parser.parse_args()

    bundle = Path(args.bundle)
    _require_file(bundle)

    extract_dir = Path(args.extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(extract_dir)

    adapter_src = extract_dir / "results" / "grpo_run_accuracy" / "lora_adapter"
    if not adapter_src.is_dir():
        raise FileNotFoundError(f"Adapter folder missing in bundle: {adapter_src}")

    adapter_dst = Path(args.runtime_adapter_dir)
    if adapter_dst.exists():
        shutil.rmtree(adapter_dst)
    adapter_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_src, adapter_dst)

    run_result_src = extract_dir / "results" / "grpo_gpu_run_result.json"
    _require_file(run_result_src)
    with run_result_src.open("r", encoding="utf-8") as f:
        run_result = json.load(f)

    if args.copy_dataset:
        dataset_src = extract_dir / "training" / "sft_data.jsonl"
        _require_file(dataset_src)
        dataset_dst = Path("training/sft_data.jsonl")
        dataset_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_src, dataset_dst)

    summary = {
        "bundle": str(bundle.resolve()),
        "extract_dir": str(extract_dir.resolve()),
        "runtime_adapter_dir": str(adapter_dst.resolve()),
        "records_used": run_result.get("records_used"),
        "train_loss": (run_result.get("metrics") or {}).get("train_loss"),
        "train_runtime_sec": (run_result.get("metrics") or {}).get("train_runtime"),
        "copy_dataset": bool(args.copy_dataset),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
