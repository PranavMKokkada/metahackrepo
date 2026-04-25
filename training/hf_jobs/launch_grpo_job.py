"""Emit a ready-to-run HF Jobs CLI command from local config."""

from __future__ import annotations

import json
import os


def main() -> None:
    recipe_path = os.path.join("results", "grpo_gpu_recipe.json")
    if not os.path.exists(recipe_path):
        print("Recipe not found. Run: python training/grpo_train.py --mode grpo")
        return

    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    command = recipe.get("hf_jobs_command", "")
    out_path = os.path.join("results", "hf_jobs_launch_command.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(command + "\n")

    print("HF Jobs launch command:")
    print(command)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
