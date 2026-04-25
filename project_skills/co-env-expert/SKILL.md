---
name: co-env-expert
description: Expert guidance for managing the CodeOrganismVM environment logic, including fault injection, reward systems (R1–R5), vitality mechanics, and codebase simulation. Use when modifying core dynamics in environment.py or simulation logic in data.py.
---

# CodeOrganismVM Environment Expert

This skill provides expert guidance for maintaining and extending the core hostile execution environment of CodeOrganismVM.

## Core Components

- **environment.py**: The central VM logic, managing vitality, action costs, and the step loop.
- **data.py**: The procedural codebase simulator and fault injector.
- **models.py**: Pydantic schemas for actions, observations, and state.

## Key Workflows

### 1. Modifying Rewards (R1–R5)
- Always refer to [rewards.md](references/rewards.md) for spec-exact weights and anti-hacking mechanisms.
- When updating `_compute_reward()` in `environment.py`, ensure the total weight sums to 1.0 (excluding hard penalties).
- **R1 (35%)**: Vitality Delta.
- **R2 (30%)**: Test Recovery.
- **R3 (15%)**: Action Efficiency.
- **R4 (10%)**: Coordination.
- **R5 (10%)**: Generalization.

### 2. Extending Fault Types
- Refer to [faults.md](references/faults.md) for the existing fault catalog and injection logic.
- Faults are phase-aware (Phase 1: Single, Phase 2: Multi, Phase 3: Adversarial).
- When adding a fault to `CodebaseSimulator` in `data.py`, ensure it is verifiable via test output.

### 3. Adjusting Vitality Mechanics
- Vitality is the organism's health [0–100].
- **Costs**: Every action has a fixed cost (e.g., `patch_file` costs -2.0).
- **Recovery**: Vitality is gained via "Metabolic Recovery" based on the ratio of passing tests in `step()`.
- **Termination**: Vitality <= 0 triggers `organism_death`.

## Best Practices
- **Watchdog First**: Always validate actions against the Watchdog layer before execution to prevent sandbox escapes or protected file writes.
- **Immutability**: Never allow the agent to modify the `/tests/` directory or its own vitality score directly.
- **Atomic State**: Use transaction locks if introducing asynchronous state mutations.
