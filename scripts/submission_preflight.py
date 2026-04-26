#!/usr/bin/env python3
"""Finale submission readiness checker for OpenEnv hackathon requirements."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests


ROOT = Path(__file__).resolve().parents[1]


def _exists(path: str) -> bool:
    return (ROOT / path).exists()


def _http_ok(url: str) -> Tuple[bool, str]:
    try:
        resp = requests.get(url, timeout=20)
        return (200 <= resp.status_code < 400), str(resp.status_code)
    except Exception as exc:  # pragma: no cover
        return False, f"error: {exc}"


def main() -> None:
    checks: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    # Local required artifacts/files
    required_files = [
        "openenv.yaml",
        "README.md",
        "training/grpo_train.py",
        "CodeOrganismVM_Training.ipynb",
        "results/eval_summary.json",
        "results/reward_curve.png",
        "results/baseline_vs_agent.png",
        "results/survival_by_phase.png",
        "results/action_distribution.png",
        "DEMO_PITCH_SLIDES.md",
    ]
    for rel in required_files:
        add(f"file:{rel}", _exists(rel), "present" if _exists(rel) else "missing")

    # README content checks
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_snippets = [
        "https://huggingface.co/spaces/teletubbies/autonomous-sre",
        "training/grpo_train.py",
        "results/eval_summary.json",
        "DEMO_PITCH_SLIDES.md",
    ]
    for snippet in required_snippets:
        ok = snippet in readme
        add(f"readme_contains:{snippet}", ok, "found" if ok else "not found")

    # Placeholder reminder checks (non-blocking, but surfaced)
    placeholders = ["ADD_YOUR_HF_BLOG_URL", "ADD_YOUR_YOUTUBE_URL"]
    for token in placeholders:
        present = token in readme
        add(
            f"placeholder:{token}",
            not present,
            "replace before submission" if present else "filled/removed",
        )

    # Hosted endpoint checks
    endpoints = {
        "space_page": "https://huggingface.co/spaces/teletubbies/autonomous-sre",
        "space_health": "https://teletubbies-autonomous-sre.hf.space/health",
        "space_ui": "https://teletubbies-autonomous-sre.hf.space/ui",
        "model_repo": "https://huggingface.co/teletubbies/autonomous-sre-lora",
        "dataset_repo": "https://huggingface.co/datasets/teletubbies/autonomous-sre-logs",
    }
    for name, url in endpoints.items():
        ok, detail = _http_ok(url)
        add(f"hosted:{name}", ok, detail)

    # Score
    hard_fail_patterns = [
        re.compile(r"^file:"),
        re.compile(r"^readme_contains:"),
        re.compile(r"^hosted:"),
    ]
    hard_checks = [
        c
        for c in checks
        if any(p.search(str(c["name"])) for p in hard_fail_patterns)
    ]
    hard_ok = sum(1 for c in hard_checks if c["ok"])
    hard_total = len(hard_checks)
    readiness_pct = round((hard_ok / max(1, hard_total)) * 100, 2)

    output = {
        "readiness_percent": readiness_pct,
        "hard_checks_ok": hard_ok,
        "hard_checks_total": hard_total,
        "checks": checks,
    }
    print(json.dumps(output, indent=2))

    if hard_ok != hard_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
