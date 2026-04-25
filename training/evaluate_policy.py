"""Deterministic policy evaluation for hackathon evidence generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import HELD_OUT_SEEDS
from training.rollout import EpisodeTrace, run_episode


PHASES = ("phase_1", "phase_2", "phase_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic policy evaluation.")
    parser.add_argument("--policies", nargs="+", default=["noop", "random", "heuristic", "stabilized", "sft"])
    parser.add_argument("--episodes-per-phase", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=11000)
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {
        "metadata": {
            "episodes_per_phase": args.episodes_per_phase,
            "phases": list(PHASES),
            "policies": args.policies,
            "held_out_seed_count": len(HELD_OUT_SEEDS),
        },
        "policies": {},
    }

    for policy in args.policies:
        traces = evaluate_policy(policy, args.episodes_per_phase, args.seed_start)
        trace_path = os.path.join(args.out_dir, f"{policy}_rollouts.jsonl")
        write_jsonl(trace_path, traces)
        summary["policies"][policy] = summarize(policy, traces)
        print(f"Wrote {len(traces)} episodes -> {trace_path}")

    summary_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {summary_path}")


def evaluate_policy(policy: str, episodes_per_phase: int, seed_start: int) -> List[EpisodeTrace]:
    traces: List[EpisodeTrace] = []
    for phase_idx, phase in enumerate(PHASES):
        seeds = fixed_seeds_for_phase(phase, episodes_per_phase, seed_start + phase_idx * 10_000)
        for seed in seeds:
            traces.append(run_episode(policy=policy, task_id=phase, seed=seed))
    return traces


def fixed_seeds_for_phase(phase: str, count: int, base_seed: int) -> List[int]:
    if phase == "phase_3" and HELD_OUT_SEEDS:
        held_out_sorted = sorted(HELD_OUT_SEEDS)
        if count <= len(held_out_sorted):
            return held_out_sorted[:count]
    return [base_seed + i for i in range(count)]


def summarize(policy: str, traces: List[EpisodeTrace]) -> Dict[str, object]:
    episode_dicts = [t.to_dict() for t in traces]
    rewards = [e["total_reward"] for e in episode_dicts]
    vitality = [e["final_vitality"] for e in episode_dicts]
    watchdog = [e["watchdog_violations"] for e in episode_dicts]
    steps = [e["steps"] for e in episode_dicts]
    recovered = [e["tests_recovered"] for e in episode_dicts]

    by_phase: Dict[str, Dict[str, float]] = {}
    for phase in PHASES:
        rows = [e for e in episode_dicts if e["task_id"] == phase]
        if not rows:
            continue
        by_phase[phase] = {
            "episodes": len(rows),
            "survival_rate": round(sum(1 for r in rows if r["survived"]) / len(rows), 4),
            "thrival_rate": round(sum(1 for r in rows if r["thrived"]) / len(rows), 4),
            "mean_reward": round(statistics.fmean(r["total_reward"] for r in rows), 4),
            "mean_final_vitality": round(statistics.fmean(r["final_vitality"] for r in rows), 4),
        }

    return {
        "policy": policy,
        "episodes": len(episode_dicts),
        "survival_rate": round(sum(1 for e in episode_dicts if e["survived"]) / max(1, len(episode_dicts)), 4),
        "thrival_rate": round(sum(1 for e in episode_dicts if e["thrived"]) / max(1, len(episode_dicts)), 4),
        "mean_reward": round(statistics.fmean(rewards), 4) if rewards else 0.0,
        "median_reward": round(statistics.median(rewards), 4) if rewards else 0.0,
        "mean_final_vitality": round(statistics.fmean(vitality), 4) if vitality else 0.0,
        "mean_watchdog_violations": round(statistics.fmean(watchdog), 4) if watchdog else 0.0,
        "mean_steps": round(statistics.fmean(steps), 4) if steps else 0.0,
        "mean_tests_recovered": round(statistics.fmean(recovered), 4) if recovered else 0.0,
        "by_phase": by_phase,
    }


def write_jsonl(path: str, traces: List[EpisodeTrace]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")


if __name__ == "__main__":
    main()
