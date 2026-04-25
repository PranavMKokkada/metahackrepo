"""Lightweight SFT-style policy tuning via rollout search.

This script trains a simple policy parameter (test cadence) by maximizing
mean reward on training seeds. Output is saved to training/policies/sft_policy.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import fmean
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rollout import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight SFT policy parameters.")
    parser.add_argument("--episodes-per-phase", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=21000)
    parser.add_argument("--interval-min", type=int, default=2)
    parser.add_argument("--interval-max", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = list(range(args.interval_min, args.interval_max + 1))
    scores: Dict[int, float] = {}

    for interval in candidates:
        rewards = evaluate_interval(interval, args.episodes_per_phase, args.seed_start)
        scores[interval] = fmean(rewards) if rewards else -999.0

    best_interval = max(scores, key=scores.get)
    payload = {
        "method": "rollout_search",
        "objective": "maximize_mean_reward",
        "episodes_per_phase": args.episodes_per_phase,
        "candidate_intervals": candidates,
        "score_by_interval": {str(k): round(v, 4) for k, v in scores.items()},
        "test_interval": int(best_interval),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "policies")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sft_policy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote trained policy -> {out_path}")
    print(f"Selected test_interval={best_interval} with mean_reward={scores[best_interval]:.4f}")


def evaluate_interval(interval: int, episodes_per_phase: int, seed_start: int) -> List[float]:
    rewards: List[float] = []
    phases = ("phase_1", "phase_2")
    for phase_idx, phase in enumerate(phases):
        base = seed_start + (phase_idx * 1000)
        for offset in range(episodes_per_phase):
            seed = base + offset
            trace = run_episode(
                policy=f"stabilized:{interval}",
                task_id=phase,
                seed=seed,
            )
            rewards.append(trace.total_reward)
    return rewards


if __name__ == "__main__":
    main()
