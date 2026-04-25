# Results Artifacts

This directory contains reproducible evaluation outputs generated from real environment rollouts.

## Generation Commands

From repository root:

```bash
python training/train_sft.py --episodes-per-phase 8
python training/evaluate_policy.py --policies noop random heuristic stabilized sft --episodes-per-phase 6 --out-dir results
python training/plot_results.py --results-dir results --summary results/eval_summary.json
python evaluation/run_eval.py
python training/grpo_train.py --mode grpo
```

## Artifact Index

- `eval_summary.json`: aggregated metrics by policy and phase.
- `noop_rollouts.jsonl`: per-episode logs for no-op baseline.
- `random_rollouts.jsonl`: per-episode logs for random baseline.
- `heuristic_rollouts.jsonl`: per-episode logs for heuristic policy.
- `stabilized_rollouts.jsonl`: per-episode logs for stabilized policy.
- `sft_rollouts.jsonl`: per-episode logs for SFT-style trained policy.
- `reward_curve.png`: running mean reward across episodes.
- `baseline_vs_agent.png`: mean reward comparison by policy.
- `survival_by_phase.png`: phase-wise survival rates.
- `action_distribution.png`: normalized action usage by policy.
- `../evaluation/report.md`: held-out and overall evaluation report.
- `notebook_training_metrics.json`: extracted training logs from a local notebook (e.g. `Untitled20.ipynb`, not committed).
- `notebook_training_curve.png`: plotted notebook loss curve.
- `training_run_summary.json`: training entrypoint run summary.

## Notes

- `phase_3` uses held-out seeds from `evaluation/held_out_seeds.json` when available.
- These metrics represent policy evaluation, not GRPO fine-tuning results.
