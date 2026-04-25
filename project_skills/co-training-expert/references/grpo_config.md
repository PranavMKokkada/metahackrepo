# GRPO Configuration for CodeOrganismVM

Group Relative Policy Optimization (GRPO) is used to optimize the survival instinct of the organism.

## Hyperparameters (Baseline)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `num_generations` | 4-8 | Group size for relative advantage. |
| `learning_rate` | 1e-5 | Cosine warmup recommended. |
| `kl_coef` | 0.02 | KL divergence penalty. |
| `max_prompt_len` | 1024 | Observation history window. |
| `max_completion_len` | 1024 | Multi-step action sequence length. |

## Reward Function Mapping
In `grpo_train.py`, each `reward_funcs` callback should map to one of the R1–R5 signals from the environment:
- `reward_vitality`: Maps to R1 (Vitality Delta).
- `reward_recovery`: Maps to R2 (Test Recovery).
- `reward_efficiency`: Maps to R3 (Step count penalty).

## Infrastructure Requirements
- **Hardware**: Single A100 (40GB) recommended for Qwen2.5-7B with Unsloth QLoRA.
- **Quantization**: 4-bit NF4 for memory efficiency.
- **Checkpoints**: Save periodically; evaluate on held-out seeds every 50 episodes.
