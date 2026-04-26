#!/usr/bin/env python3
"""One-shot local smoke test for base model + LoRA adapter."""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    base_model = os.getenv("BASE_MODEL_ID", "unsloth/llama-3-8b-Instruct-bnb-4bit")
    adapter_path = os.getenv("LORA_ADAPTER_PATH", "artifacts/runtime/lora_adapter")
    prompt = os.getenv(
        "SMOKE_PROMPT",
        'Return a single JSON action for an SRE agent. Observation: failing test in src/auth.py. '
        'Valid action example: {"action_type":"run_tests"}',
    )
    max_new_tokens = int(os.getenv("SMOKE_MAX_NEW_TOKENS", "96"))

    try:
        import torch  # type: ignore
        from peft import AutoPeftModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Missing inference deps: {exc}"}))
        sys.exit(1)

    if not os.path.isdir(adapter_path):
        print(json.dumps({"ok": False, "error": f"Adapter path not found: {adapter_path}"}))
        sys.exit(1)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        base_model_name_or_path=base_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(json.dumps({"ok": True, "base_model": base_model, "adapter_path": adapter_path, "output": text}))


if __name__ == "__main__":
    main()
