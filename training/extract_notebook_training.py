"""Extract training logs from a notebook and generate summary artifacts.

Parses notebook output lines like:
{'loss': '0.123', 'grad_norm': '0.45', 'learning_rate': '1e-4', 'epoch': '0.25'}
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from statistics import fmean
from typing import Any, Dict, List


LINE_PATTERN = re.compile(r"\{'loss':\s*'[^']+',\s*'grad_norm':\s*'[^']+',\s*'learning_rate':\s*'[^']+',\s*'epoch':\s*'[^']+'\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract notebook training metrics.")
    parser.add_argument("--notebook", default="Untitled20.ipynb")
    parser.add_argument("--out-json", default="results/notebook_training_metrics.json")
    parser.add_argument("--out-plot", default="results/notebook_training_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.notebook, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    rows = extract_rows(notebook)
    if not rows:
        raise SystemExit("No notebook training rows found.")

    summary = summarize_rows(rows)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"Wrote {args.out_json}")

    make_plot(rows, args.out_plot)
    print(f"Wrote {args.out_plot}")


def extract_rows(notebook: Dict[str, Any]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for cell in notebook.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text", [])
            if isinstance(text, str):
                text = [text]
            for line in text:
                for match in LINE_PATTERN.findall(line):
                    parsed = ast.literal_eval(match)
                    rows.append(
                        {
                            "loss": float(parsed["loss"]),
                            "grad_norm": float(parsed["grad_norm"]),
                            "learning_rate": float(parsed["learning_rate"]),
                            "epoch": float(parsed["epoch"]),
                        }
                    )
    return rows


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    losses = [r["loss"] for r in rows]
    return {
        "steps_logged": float(len(rows)),
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "best_loss": round(min(losses), 6),
        "mean_loss": round(fmean(losses), 6),
        "loss_delta": round(losses[-1] - losses[0], 6),
    }


def make_plot(rows: List[Dict[str, float]], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    x = [r["epoch"] for r in rows]
    y = [r["loss"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.7)
    plt.title("Notebook Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
