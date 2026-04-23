"""Task definitions and curriculum for CodeOrganismVM (spec §7.3).

Three phases of increasing hostility:
  phase_1 — Single fault, 20 steps, N=8 interval. Gate: ≥30% survival.
  phase_2 — Multi-fault (2–4), 50 steps, N=6 interval. Gate: ≥55% survival.
  phase_3 — Adversarial adaptive, 100 steps, N=4 interval. Gate: ≥40% held-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from models import Action
from environment import CodeOrganismEnv


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    fault_interval: int
    initial_faults: int
    advancement_gate: str
    scoring_summary: str


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "phase_1": TaskDefinition(
        task_id="phase_1",
        name="Phase 1: Metabolic Stabilization",
        description=(
            "Single fault per episode. Agent must detect and patch a corrupted "
            "import, flipped assertion, missing env var, null return, or off-by-one. "
            "Focus: basic fault diagnosis and healing."
        ),
        difficulty="easy",
        max_steps=20,
        fault_interval=8,
        initial_faults=1,
        advancement_gate="Episode survival ≥ 30%",
        scoring_summary="R1 vitality=35%, R2 tests=30%, R3 efficiency=15%, R4 coord=10%, R5 novel=10%",
    ),
    "phase_2": TaskDefinition(
        task_id="phase_2",
        name="Phase 2: Colony Survival",
        description=(
            "Multi-fault environment (2–4 initial). Fault injector fires every 6 steps. "
            "Subagent spawning needed for parallel repair. Phase 2 adds: dependency_cycle, "
            "permission_revoked, race_condition, schema_mismatch."
        ),
        difficulty="medium",
        max_steps=50,
        fault_interval=6,
        initial_faults=3,
        advancement_gate="Survival ≥ 55%",
        scoring_summary="R1 vitality=35%, R2 tests=30%, R3 efficiency=15%, R4 coord=10%, R5 novel=10%",
    ),
    "phase_3": TaskDefinition(
        task_id="phase_3",
        name="Phase 3: Adversarial Expert",
        description=(
            "FaultInjector adapts: targets agent's last-patched modules. Adds: "
            "targeted_regression, cascade_corruption, checkpoint_invalidation. "
            "Expert validators challenge patch quality. High-stress survival."
        ),
        difficulty="expert",
        max_steps=100,
        fault_interval=4,
        initial_faults=4,
        advancement_gate="Held-out seed success ≥ 40%",
        scoring_summary="R1 vitality=35%, R2 recovery=30%, R3 efficiency=15%, R4 coord=10%, R5 novelty=10%",
    ),
}


def run_grader(task_id: str, actions: List[dict]) -> dict:
    """Replay an action sequence and grade the episode."""
    env = CodeOrganismEnv()
    obs = env.reset(task_id)
    per_step: List[dict] = []
    total_reward = 0.0
    watchdog_total = 0

    for action_dict in actions:
        try:
            action = Action(**action_dict)
        except Exception as e:
            per_step.append({"error": str(e)})
            continue

        result = env.step(action)
        per_step.append({
            "timestep": len(per_step),
            "reward": result.reward,
            "breakdown": result.reward_breakdown.model_dump(),
            "vitality": round(env._vitality, 2),
        })
        total_reward += result.reward
        watchdog_total += env._watchdog_violations

        if result.done:
            break

    steps_taken = max(1, len(per_step))
    return {
        "task_id": task_id,
        "score": round(max(0.01, min(total_reward / steps_taken, 1.0)), 4),
        "total_reward": round(total_reward, 4),
        "steps_taken": steps_taken,
        "final_vitality": round(env._vitality, 2),
        "survived": env._vitality > 0,
        "thrived": env._thriving_streak >= 3 and env._vitality > 80,
        "watchdog_violations": watchdog_total,
        "faults_injected": len(env._simulator.faults) if env._simulator else 0,
        "per_step": per_step,
    }
