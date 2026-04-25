---
name: co-training-expert
description: Expert guidance for the CodeOrganismVM training pipeline, including SFT data generation, GRPO RL configuration, and Gym environment wrapping. Use when optimizing model performance or debugging training loops.
---

# CodeOrganismVM Training Orchestrator

This skill provides expert guidance for training LLM agents to survive the CodeOrganismVM environment using SFT and GRPO.

## Core Components

- **training/generate_sft_data.py**: Script for creating synthetic expert traces to warm-start the model.
- **training/grpo_train.py**: The main RL training loop using TRL's `GRPOTrainer`.
- **gym_wrapper.py**: Gymnasium-compatible wrapper for the CodeOrganismVM environment.

## Key Workflows

### 1. Generating SFT Traces
- Traces are used to teach the model **format compliance** (JSON actions), not optimal strategies.
- Ensure `sft_data.jsonl` contains a mix of all 7 action types.
- To add new traces, update the `trace_templates` in `generate_sft_data.py`.

### 2. Configuring GRPO RL
- Refer to [grpo_config.md](references/grpo_config.md) for hyperparameter details and reward function mapping.
- **Group Size**: GRPO requires multiple rollouts (e.g., 4 or 8) per prompt to compute relative advantage.
- **KL Coefficient**: Keep low (0.01-0.02) to encourage early exploration.

### 3. Debugging Training
- **Reward Collapse**: If reward stays negative, check if the fault interval is too short for the current model capacity.
- **KL Divergence**: If KL spikes, the model is diverging too fast; reduce the learning rate or increase the KL penalty.

## Best Practices
- **Warm Start**: Always perform an SFT pass before starting GRPO to ensure the model can at least call the available tools.
- **Curriculum Gates**: Only advance the training curriculum phase (P1 -> P2 -> P3) once the survival threshold is met.
- **Seed Registry**: Use the seed registry to ensure training and evaluation distributions are disjoint.
