"""Generate a concise held-out evaluation report from results/eval_summary.json."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def main() -> None:
    results_dir = "results"
    summary_path = os.path.join(results_dir, "eval_summary.json")
    out_path = os.path.join("evaluation", "report.md")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary: Dict[str, Any] = json.load(f)

    policies = summary.get("policies", {})
    lines = [
        "# Evaluation Report",
        "",
        f"- Episodes per phase: {summary.get('metadata', {}).get('episodes_per_phase', 'n/a')}",
        f"- Held-out seed count: {summary.get('metadata', {}).get('held_out_seed_count', 'n/a')}",
        "",
        "## Held-out (Phase 3) Snapshot",
        "",
        "| Policy | Phase 3 Survival | Phase 3 Mean Reward | Phase 3 Mean Final Vitality |",
        "|---|---:|---:|---:|",
    ]

    for policy, metrics in policies.items():
        phase3 = metrics.get("by_phase", {}).get("phase_3", {})
        lines.append(
            f"| {policy} | {phase3.get('survival_rate', 0):.4f} | "
            f"{phase3.get('mean_reward', 0):.4f} | {phase3.get('mean_final_vitality', 0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Overall Snapshot",
            "",
            "| Policy | Survival Rate | Mean Reward | Mean Final Vitality | Mean Steps |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for policy, metrics in policies.items():
        lines.append(
            f"| {policy} | {metrics.get('survival_rate', 0):.4f} | "
            f"{metrics.get('mean_reward', 0):.4f} | {metrics.get('mean_final_vitality', 0):.4f} | "
            f"{metrics.get('mean_steps', 0):.4f} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
