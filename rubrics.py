"""
Composable Reward Rubrics for Autonomous SRE Control Center.
Implements the R1–R5 SRE Rubric as per judging criteria.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any
from models import TestResult, Action, CodeOrganismActionType, RewardBreakdown

class SRERubricScorer:
    """Aggregates multiple composable rubrics into a final reward signal."""

    def compute(
        self, 
        action: Action, 
        current_tests: List[TestResult], 
        prev_vitality: float, 
        current_vitality: float,
        action_count: int,
        action_history: List[str],
        active_intent: str | None,
        is_done: bool,
        is_held_out: bool,
        phase_num: int,
        watchdog_penalty: float
    ) -> RewardBreakdown:
        
        # R1: SLA Stability (Vitality Delta)
        vitality_delta = current_vitality - prev_vitality
        r1 = max(-1.0, min(1.0, vitality_delta / 10.0))
        
        # R2: Recovery Success (Test Recovery)
        r2 = 0.0
        recovered_count = 0
        degraded_count = 0
        for t in current_tests:
            if t.delta == 1:
                r2 += 1.0  # FAIL -> PASS
                recovered_count += 1
            elif t.delta == -1:
                r2 -= 0.5  # PASS -> FAIL
                degraded_count += 1
            
        # R3: Remediation Efficiency (1/sqrt(n))
        r3 = 1.0 / math.sqrt(max(1, action_count))
        if len(action_history) >= 2 and action_history[-1] == action_history[-2]:
            r3 -= 0.3  # Duplicate penalty
        if action.action_type in (CodeOrganismActionType.DO_NOTHING, CodeOrganismActionType.EMIT_SIGNAL):
            r3 -= 0.2
        if len(action_history) >= 3 and action_history[-3:] == [
            CodeOrganismActionType.EMIT_SIGNAL.value,
            CodeOrganismActionType.EMIT_SIGNAL.value,
            CodeOrganismActionType.EMIT_SIGNAL.value,
        ]:
            r3 -= 0.6
            
        # R4: Teaming Bonus (Intent signaling & Delegation)
        r4 = 0.0
        if action.action_type == CodeOrganismActionType.PATCH_FILE:
            if active_intent == action.path:
                r4 += 0.5  # Planning bonus
            if len(action_history) >= 2 and action_history[-2] == CodeOrganismActionType.EMIT_SIGNAL.value:
                r4 += 0.2
        if action.action_type == CodeOrganismActionType.EMIT_SIGNAL and recovered_count == 0:
            r4 -= 0.2
        # (Subagent coordination r4 is handled in environment.py _handle_subagent)

        # Reward useful action-outcome chains and penalize low-value loops.
        if action.action_type == CodeOrganismActionType.RUN_TESTS and recovered_count > 0:
            r2 += 0.25 * recovered_count
        if action.action_type == CodeOrganismActionType.ROLLBACK and degraded_count > 0:
            r2 += 0.2
        if action.action_type == CodeOrganismActionType.QUARANTINE and degraded_count == 0 and recovered_count == 0:
            r2 -= 0.25
        
        # R5: Architecture Generalization (Held-out seeds)
        r5 = 0.0
        if is_done and current_vitality > 0:
            if is_held_out:
                r5 = 0.5
            elif phase_num == 3:
                r5 = 0.2

        total = (
            0.35 * r1 +
            0.30 * r2 +
            0.15 * r3 +
            0.10 * r4 +
            0.10 * r5 +
            watchdog_penalty
        )

        return RewardBreakdown(
            vitality_delta=round(r1, 4),
            test_recovery=round(r2, 4),
            efficiency_bonus=round(r3, 4),
            coordination_bonus=round(r4, 4),
            novelty_bonus=round(r5, 4),
            watchdog_penalty=round(watchdog_penalty, 4),
            total=round(total, 4),
        )
