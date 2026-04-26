#!/usr/bin/env python3
"""Check whether local hardware is enough for base+LoRA inference."""

from __future__ import annotations

import json
import platform
from typing import Any, Dict


def _check() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": False,
        "can_run_local_lora": False,
        "recommendation": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        report["recommendation"] = f"Install torch first ({exc})."
        return report

    report["cuda_available"] = bool(torch.cuda.is_available())
    if not report["cuda_available"]:
        report["recommendation"] = (
            "No CUDA GPU detected. Use Hugging Face Inference endpoint or "
            "run local API without loading base+LoRA."
        )
        return report

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_gb = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2)
    report["gpu_name"] = name
    report["gpu_total_vram_gb"] = total_gb

    # 8B 4-bit + LoRA typically needs >= 16GB VRAM.
    if total_gb >= 16:
        report["can_run_local_lora"] = True
        report["recommendation"] = "Local base+LoRA inference is feasible on this GPU."
    else:
        report["recommendation"] = (
            "GPU VRAM is likely insufficient for stable local base+LoRA inference. "
            "Prefer remote inference deployment."
        )
    return report


def main() -> None:
    print(json.dumps(_check(), indent=2))


if __name__ == "__main__":
    main()
