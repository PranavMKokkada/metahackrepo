"""Generate result plots from committed rollout logs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--summary", default="results/eval_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install training dependencies first.") from exc

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    os.makedirs(args.results_dir, exist_ok=True)
    rollouts = load_rollouts(args.results_dir, summary["metadata"]["policies"])

    make_reward_curve_plot(plt, rollouts, os.path.join(args.results_dir, "reward_curve.png"))
    make_baseline_plot(plt, summary["policies"], os.path.join(args.results_dir, "baseline_vs_agent.png"))
    make_survival_by_phase_plot(plt, summary["policies"], os.path.join(args.results_dir, "survival_by_phase.png"))
    make_action_distribution_plot(plt, rollouts, os.path.join(args.results_dir, "action_distribution.png"))
    print("Generated reward_curve.png, baseline_vs_agent.png, survival_by_phase.png, action_distribution.png")


def load_rollouts(results_dir: str, policies: List[str]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for policy in policies:
        path = os.path.join(results_dir, f"{policy}_rollouts.jsonl")
        rows: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[policy] = rows
    return out


def make_reward_curve_plot(plt, rollouts: Dict[str, List[Dict[str, object]]], out_path: str) -> None:
    plt.figure(figsize=(10, 5))
    for policy, episodes in rollouts.items():
        running = []
        total = 0.0
        for idx, ep in enumerate(episodes, start=1):
            total += float(ep["total_reward"])
            running.append(total / idx)
        plt.plot(range(1, len(running) + 1), running, label=policy)
    plt.title("Running Mean Reward by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Running Mean Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_baseline_plot(plt, policies: Dict[str, Dict[str, object]], out_path: str) -> None:
    labels = list(policies.keys())
    rewards = [float(policies[p]["mean_reward"]) for p in labels]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, rewards)
    plt.title("Mean Reward Comparison")
    plt.xlabel("Policy")
    plt.ylabel("Mean Episode Reward")
    for bar, value in zip(bars, rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_survival_by_phase_plot(plt, policies: Dict[str, Dict[str, object]], out_path: str) -> None:
    phases = ["phase_1", "phase_2", "phase_3"]
    x = range(len(phases))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for idx, (policy, metrics) in enumerate(policies.items()):
        rates = []
        by_phase = metrics.get("by_phase", {})
        for phase in phases:
            phase_info = by_phase.get(phase, {})
            rates.append(float(phase_info.get("survival_rate", 0.0)))
        shifted = [v + (idx - 1) * width for v in x]
        plt.bar(shifted, rates, width=width, label=policy)

    plt.xticks(list(x), phases)
    plt.ylim(0, 1.05)
    plt.title("Survival Rate by Phase")
    plt.xlabel("Phase")
    plt.ylabel("Survival Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_action_distribution_plot(plt, rollouts: Dict[str, List[Dict[str, object]]], out_path: str) -> None:
    counters = defaultdict(Counter)
    action_names = set()

    for policy, episodes in rollouts.items():
        for episode in episodes:
            for action in episode.get("actions", []):
                name = str(action.get("action_type", "unknown"))
                counters[policy][name] += 1
                action_names.add(name)

    actions_sorted = sorted(action_names)
    x = range(len(actions_sorted))
    width = 0.25

    plt.figure(figsize=(11, 5))
    for idx, policy in enumerate(counters.keys()):
        total = sum(counters[policy].values()) or 1
        values = [counters[policy][a] / total for a in actions_sorted]
        shifted = [v + (idx - 1) * width for v in x]
        plt.bar(shifted, values, width=width, label=policy)

    plt.xticks(list(x), actions_sorted, rotation=25, ha="right")
    plt.title("Action Distribution by Policy")
    plt.xlabel("Action Type")
    plt.ylabel("Action Share")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
