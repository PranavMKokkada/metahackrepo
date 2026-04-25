"""Core CodeOrganismVM environment (spec §4, §5, §6, §9.3).

Implements step() / reset() / state() for an agent living in a hostile,
self-corrupting codebase. Features:
  - Spec-exact vitality costs per action
  - Auto-checkpointing every 5 steps
  - Watchdog security layer with penalty logging
  - Snorkel AI expert validation via request_expert()
  - Phase-aware fault injection (P1: N=8, P2: N=6, P3: N=4 + adaptive)
  - R1–R5 reward computation with anti-hacking protections
  - Quarantine mechanics with overcorrection tax
  - Rollback loop protection (3 per checkpoint)
  - Thrival: all_pass for 3 consecutive steps AND vitality > 80
  - World Modeling: dependency_graph in observations
  - Multi-Agent Teaming: Functional signaling with coordination bonuses
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from secrets import SystemRandom
from typing import Optional, Dict, List, Any

# Standard OpenEnv Core (spec §30)
try:
    from openenv_core import Environment
except ImportError:
    class Environment: pass

from models import (
    Action,
    CodeOrganismActionType,
    Observation,
    RewardBreakdown,
    StepResult,
    EnvState,
    TestResult,
    Checkpoint,
    SubagentResult,
    ExpertResponse,
)
from data import CodebaseSimulator, get_curriculum_seed, is_protected_path, HELD_OUT_SEEDS
from rubrics import SRERubricScorer

# ── Vitality Costs (spec §4.2 — exact values) ─────────────────────────────────

VITALITY_COSTS: Dict[CodeOrganismActionType, float] = {
    CodeOrganismActionType.PATCH_FILE:      2.0,   # −2
    CodeOrganismActionType.RUN_TESTS:       3.0,   # −3
    CodeOrganismActionType.SPAWN_SUBAGENT:  5.0,   # −5
    CodeOrganismActionType.QUARANTINE:      1.0,   # −1
    CodeOrganismActionType.ROLLBACK:        4.0,   # −4
    CodeOrganismActionType.REQUEST_EXPERT:  6.0,   # −6
    CodeOrganismActionType.EMIT_SIGNAL:     0.0,   #  0
    CodeOrganismActionType.DO_NOTHING:      0.0,   #  0
}

# ── Phase-aware configuration (spec §4.3, §7.3) ───────────────────────────────

PHASE_CONFIG = {
    "phase_1": {"max_steps": 20,  "fault_interval": 8, "initial_faults": 1, "phase_num": 1},
    "phase_2": {"max_steps": 50,  "fault_interval": 6, "initial_faults": 3, "phase_num": 2},
    "phase_3": {"max_steps": 100, "fault_interval": 4, "initial_faults": 4, "phase_num": 3},
}

INCIDENT_SCENARIO_CARDS = {
    1: "SEV-2 API latency spike after minor config drift; objective: restore baseline health quickly.",
    2: "SEV-1 cascading service degradation across auth/queue/cache paths; objective: contain blast radius.",
    3: "ADVERSARIAL regression campaign targeting recent fixes; objective: survive while preserving safety.",
}

# ── Watchdog penalties (spec §6.2) ─────────────────────────────────────────────

WATCHDOG_PROTECTED_FILE_PENALTY = -10.0
WATCHDOG_ENV_SCOPE_PENALTY = -3.0
WATCHDOG_BAD_TOOL_PENALTY = -10.0
WATCHDOG_ESCAPE_PENALTY = -15.0

AUTO_CHECKPOINT_INTERVAL = 5   # Spec: every 5 steps
MAX_QUARANTINED_FREE = 3       # Spec §10: >3 quarantines → tax
RUNTIME_RNG = SystemRandom()


class CodeOrganismEnv:
    """The hostile virtual machine environment (spec §4)."""

    def __init__(self) -> None:
        self._task_id: str = "phase_1"
        self._phase_num: int = 1
        self._simulator: Optional[CodebaseSimulator] = None
        self._step: int = 0
        self._vitality: float = 100.0
        self._prev_vitality: float = 100.0
        self._max_steps: int = 20
        self._fault_interval: int = 8
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._reward_history: List[float] = []
        self._thriving_streak: int = 0
        self._last_test_results: List[TestResult] = []
        self._episode_id: int = 0
        self._action_count: int = 0
        self._action_history: List[str] = []  # For duplicate detection (R3)
        self._watchdog_flags: List[str] = []
        self._watchdog_violations: int = 0
        self._pending_subagent_results: List[SubagentResult] = []
        self._signals_log: List[Dict[str, Any]] = []
        
        # Hackathon Teaming & World Modeling (Functional additions)
        self._active_intent: Optional[str] = None
        self._step_alerts: List[str] = []
        
        # SRE Industry Pivot Metrics
        self._total_downtime_saved: float = 0.0 # in seconds
        self._last_action_confidence: float = 0.0
        self._last_action_risk: str = "Low"
        self._scorer = SRERubricScorer()

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self, task_id: str = "phase_1", seed: Optional[int] = None) -> Observation:
        """Generate a fresh broken codebase (spec §4.4)."""
        self._task_id = task_id
        cfg = PHASE_CONFIG.get(task_id, PHASE_CONFIG["phase_1"])
        self._phase_num = cfg["phase_num"]
        self._max_steps = cfg["max_steps"]
        self._fault_interval = cfg["fault_interval"]

        if seed is None:
            self._episode_id = RUNTIME_RNG.randint(0, 100000)
            simulator_seed = get_curriculum_seed(self._phase_num, self._episode_id)
        else:
            simulator_seed = int(seed)
            self._episode_id = simulator_seed
        self._simulator = CodebaseSimulator(simulator_seed, phase=self._phase_num)

        # Initial faults
        for _ in range(cfg["initial_faults"]):
            self._simulator.inject_fault(step=0, phase=self._phase_num)

        self._step = 0
        self._vitality = 100.0
        self._prev_vitality = 100.0
        self._done = False
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._thriving_streak = 0
        self._action_count = 0
        self._action_history = []
        self._watchdog_flags = []
        self._watchdog_violations = 0
        self._pending_subagent_results = []
        self._signals_log = []
        
        # Reset teaming and SRE state
        self._active_intent = None
        self._step_alerts = []
        self._total_downtime_saved = 0.0
        self._last_action_confidence = 0.0
        self._last_action_risk = "Low"
        scenario = INCIDENT_SCENARIO_CARDS.get(self._phase_num, INCIDENT_SCENARIO_CARDS[1])
        self._step_alerts = [f"INCIDENT_CARD: {scenario}"]

        # Auto-checkpoint at step 0
        self._simulator.create_checkpoint(self._vitality, 0)

        # Initial test run
        self._last_test_results = self._simulator.run_all_tests()

        return self._make_observation()

    def inject_chaos(self, fault_type: str = "random") -> str:
        """SRE Chaos Engineering: Manually trigger a system fault."""
        if self._done or not self._simulator:
            return "Error: System not active."
        
        if fault_type == "random":
            self._simulator.inject_fault(self._step, self._phase_num)
            msg = "Random Chaos Fault Injected."
        else:
            # We add a helper to simulator for specific types if needed, 
            # for now, random or specific if it matches catalog
            self._simulator.inject_fault(self._step, self._phase_num)
            msg = f"Chaos Engine: {fault_type} triggered."
            
        self._vitality -= 5.0
        self._step_alerts.append(f"⚠️ MANUAL CHAOS TRIGGER: {msg}")
        return msg

    def step(self, action: Action) -> StepResult:
        """Process action and advance the hostile environment (spec §4.5)."""
        if self._done:
            return StepResult(done=True, info={"error": "Episode finished."})

        self._step += 1
        self._action_count += 1
        self._prev_vitality = self._vitality
        step_watchdog_flags: List[str] = []
        watchdog_penalty = self._apply_watchdog(action, step_watchdog_flags)
        self._apply_action_costs(action, step_watchdog_flags)
        action_info = self._process_action(action)
        self._action_history.append(action.action_type.value)
        self._maybe_checkpoint()
        self._maybe_inject_periodic_fault()
        current_tests, num_passing, total_tests = self._evaluate_system_state()
        info = self._compute_termination_info()

        # ────────────────────────────────────────────────────────────────────
        # 8. Compute reward R1–R5 (spec §6)
        # ────────────────────────────────────────────────────────────────────
        breakdown = self._compute_reward(action, current_tests, watchdog_penalty)
        self._cumulative_reward += breakdown.total
        self._reward_history.append(breakdown.total)
        
        # SRE Business Metrics: Simulated Downtime Saved
        # Logic: Each FAIL->PASS test saves 300s of downtime
        recovered = sum(1 for t in current_tests if t.delta == 1)
        if recovered > 0:
            saving = recovered * 300 # 5 mins per test
            self._total_downtime_saved += saving
            self._step_alerts.append(f"✅ SRE IMPACT: Autonomous remediation saved {saving}s of system downtime.")

        # SRE Explainability Simulation
        self._last_action_confidence = (
            0.8 + (RUNTIME_RNG.random() * 0.15)
            if breakdown.total > 0
            else 0.4 + (RUNTIME_RNG.random() * 0.3)
        )
        self._last_action_risk = self._risk_from_confidence(self._last_action_confidence)

        # Record watchdog flags
        self._watchdog_flags = step_watchdog_flags
        self._last_test_results = current_tests

        obs = None if self._done else self._make_observation()

        return StepResult(
            observation=obs,
            reward=breakdown.total,
            reward_breakdown=breakdown,
            done=self._done,
            info={
                **info, 
                "action_result": action_info,
                "postmortem": self._episode_postmortem(info.get("termination", "running")) if self._done else None,
                "sre_metrics": {
                    "confidence": round(self._last_action_confidence, 2),
                    "risk_assessment": self._last_action_risk,
                    "downtime_saved_total": self._total_downtime_saved
                }
            },
        )

    def _apply_watchdog(self, action: Action, step_watchdog_flags: List[str]) -> float:
        watchdog_penalty, flags = self._watchdog_check(action)
        step_watchdog_flags.extend(flags)
        return watchdog_penalty

    def _apply_action_costs(self, action: Action, step_watchdog_flags: List[str]) -> None:
        self._vitality -= VITALITY_COSTS.get(action.action_type, 1.0)
        excess_quarantines = max(0, len(self._simulator.quarantined_modules) - MAX_QUARANTINED_FREE)
        if excess_quarantines <= 0:
            return
        tax = excess_quarantines * 2.0
        self._vitality -= tax
        step_watchdog_flags.append(f"Overcorrection tax: −{tax} vitality for {excess_quarantines} excess quarantines")

    def _maybe_checkpoint(self) -> None:
        if self._step % AUTO_CHECKPOINT_INTERVAL == 0:
            self._simulator.create_checkpoint(self._vitality, self._step)

    def _maybe_inject_periodic_fault(self) -> None:
        self._step_alerts = []
        if self._step <= 0 or self._step % self._fault_interval != 0:
            return
        if self._phase_num == 3:
            fault = self._simulator.inject_targeted_fault(self._step)
            if fault:
                self._step_alerts.append(f"🚨 NEUROTOXIN DETECTED: Adversarial mutation targeting {fault.target}")
        else:
            self._simulator.inject_fault(self._step, self._phase_num)
        self._vitality -= 5.0

    def _evaluate_system_state(self) -> tuple[List[TestResult], int, int]:
        current_tests = self._simulator.run_all_tests()
        num_passing = sum(1 for t in current_tests if t.status == "PASS")
        total_tests = max(1, len(current_tests))
        self._apply_metabolic_recovery(num_passing, total_tests)
        self._compute_test_deltas(current_tests)
        self._update_thriving_streak(num_passing, total_tests)
        return current_tests, num_passing, total_tests

    def _apply_metabolic_recovery(self, num_passing: int, total_tests: int) -> None:
        pass_ratio = num_passing / total_tests
        vitality_gain = pass_ratio * 3.0
        self._vitality = min(100.0, max(0.0, self._vitality + vitality_gain))

    def _compute_test_deltas(self, current_tests: List[TestResult]) -> None:
        prev_map = {t.name: t.status for t in self._last_test_results}
        for test in current_tests:
            prev = prev_map.get(test.name, "PASS")
            if prev != "PASS" and test.status == "PASS":
                test.delta = 1
            elif prev == "PASS" and test.status != "PASS":
                test.delta = -1
            else:
                test.delta = 0

    def _update_thriving_streak(self, num_passing: int, total_tests: int) -> None:
        if num_passing == total_tests:
            self._thriving_streak += 1
            return
        self._thriving_streak = 0

    def _compute_termination_info(self) -> Dict[str, str]:
        if self._vitality <= 0:
            self._vitality = 0
            self._done = True
            return {"termination": "organism_death"}
        if self._thriving_streak >= 3 and self._vitality > 80:
            self._done = True
            return {"termination": "organism_thrival"}
        if self._step >= self._max_steps:
            self._done = True
            return {"termination": "timeout_death"}
        return {}

    def _current_slo_metrics(self) -> Dict[str, float]:
        tests = self._last_test_results or []
        total_tests = max(1, len(tests))
        passing = sum(1 for t in tests if t.status == "PASS")
        failing = total_tests - passing
        availability = round((passing / total_tests) * 100.0, 2)
        error_rate = round((failing / total_tests) * 100.0, 2)
        p95_latency_ms = round(120.0 + (error_rate * 8.0) + max(0.0, 100.0 - self._vitality), 2)
        blast_radius = round((len(self._simulator.faults) / total_tests) * 100.0, 2) if self._simulator else 0.0
        return {
            "availability_pct": availability,
            "error_rate_pct": error_rate,
            "p95_latency_ms": p95_latency_ms,
            "blast_radius_pct": blast_radius,
            "incident_severity": round(min(100.0, error_rate + (100.0 - self._vitality)), 2),
        }

    def _episode_postmortem(self, termination: str) -> str:
        slo = self._current_slo_metrics()
        return (
            f"termination={termination}; vitality={round(self._vitality, 2)}; "
            f"availability={slo['availability_pct']}%; error_rate={slo['error_rate_pct']}%; "
            f"p95_latency_ms={slo['p95_latency_ms']}; watchdog_violations={self._watchdog_violations}; "
            f"faults_remaining={len(self._simulator.faults) if self._simulator else 0}"
        )

    def state(self) -> EnvState:
        """Current lifecycle state."""
        current_tests = self._last_test_results if self._simulator else []
        return EnvState(
            task_id=self._task_id,
            vitality=round(self._vitality, 2),
            current_step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            faults_injected=len(self._simulator.faults) if self._simulator else 0,
            tests_passing=sum(1 for t in current_tests if t.status == "PASS"),
            tests_total=len(current_tests),
            active_quarantines=list(self._simulator.quarantined_modules) if self._simulator else [],
            reward_history=self._reward_history[-20:],
            episode_id=self._episode_id,
            watchdog_violations=self._watchdog_violations,
        )

    # ── Watchdog Security Layer (spec §6.2) ────────────────────────────────

    def _watchdog_check(self, action: Action) -> tuple[float, List[str]]:
        """Enforce action boundaries. Returns (penalty, flags)."""
        penalty = 0.0
        flags: List[str] = []

        # Protected file check
        if action.action_type == CodeOrganismActionType.PATCH_FILE and action.path:
            if is_protected_path(action.path):
                penalty += WATCHDOG_PROTECTED_FILE_PENALTY
                flags.append(f"WATCHDOG: Protected file write attempt on '{action.path}' → −5 penalty")
                self._watchdog_violations += 1

        return penalty, flags

    # ── Action Handlers ────────────────────────────────────────────────────

    def _process_action(self, action: Action) -> dict:
        handler_map = {
            CodeOrganismActionType.PATCH_FILE: self._handle_patch_file_action,
            CodeOrganismActionType.RUN_TESTS: self._handle_run_tests_action,
            CodeOrganismActionType.ROLLBACK: self._handle_rollback_action,
            CodeOrganismActionType.SPAWN_SUBAGENT: self._handle_subagent,
            CodeOrganismActionType.QUARANTINE: self._handle_quarantine_action,
            CodeOrganismActionType.REQUEST_EXPERT: self._handle_expert,
            CodeOrganismActionType.EMIT_SIGNAL: self._handle_emit_signal_action,
            CodeOrganismActionType.DO_NOTHING: self._handle_do_nothing_action,
        }
        handler = handler_map.get(action.action_type)
        if handler is None:
            return {"result": "unknown_action"}
        return handler(action)

    def _handle_patch_file_action(self, action: Action) -> dict:
        if not action.path or not action.diff:
            return {"result": "error", "message": "path and diff required."}
        if is_protected_path(action.path):
            return {"result": "blocked", "message": "Watchdog: protected path."}
        ok = self._simulator.apply_patch(action.path, action.diff, self._step)
        return {"result": "success" if ok else "failure", "path": action.path}

    def _handle_run_tests_action(self, _action: Action) -> dict:
        results = self._simulator.run_all_tests()
        passing = sum(1 for t in results if t.status == "PASS")
        return {"result": "success", "tests_passing": passing, "tests_total": len(results)}

    def _handle_rollback_action(self, action: Action) -> dict:
        if not action.checkpoint_id:
            return {"result": "error", "message": "checkpoint_id required."}
        ok, msg = self._simulator.rollback(action.checkpoint_id)
        if ok:
            self._restore_vitality_from_checkpoint(action.checkpoint_id)
        return {"result": "success" if ok else "failure", "message": msg}

    def _restore_vitality_from_checkpoint(self, checkpoint_id: str) -> None:
        for checkpoint in self._simulator.checkpoints:
            if checkpoint["id"] != checkpoint_id:
                continue
            saved_vitality = checkpoint["state"].get("vitality", self._vitality)
            self._vitality = min(100.0, (self._vitality + saved_vitality) / 2)
            return

    def _handle_quarantine_action(self, action: Action) -> dict:
        module = action.module or action.path or ""
        if not module:
            return {"result": "error", "message": "module required."}
        info = self._simulator.quarantine_module(module)
        return {"result": "success", **info}

    def _handle_emit_signal_action(self, action: Action) -> dict:
        signal = {
            "type": action.signal_type or "generic",
            "data": action.signal_data or {},
            "step": self._step,
        }
        if signal["type"] == "INTENT_PATCH" and "target" in signal["data"]:
            self._active_intent = signal["data"]["target"]
        self._signals_log.append(signal)
        return {"result": "signal_emitted", "signal": signal}

    @staticmethod
    def _handle_do_nothing_action(_action: Action) -> dict:
        return {"result": "idle", "message": "Organism metabolizing."}

    def _handle_subagent(self, action: Action) -> dict:
        """Subagent simulation (spec §9.6)."""
        task_desc = action.task or "generic repair"
        sim = self._simulator

        # Spec §10 Edge Case: Subagent recursion
        if "spawn" in task_desc.lower() or "delegate" in task_desc.lower():
            result = SubagentResult(
                task=task_desc,
                success=False,
                actions_taken=1,
                tests_fixed=0,
                vitality_delta=-1.0,
                detail="Error: Subagent nesting depth exceeded (max=1). Subagent terminated.",
            )
            self._pending_subagent_results.append(result)
            return {"result": "subagent_complete", "success": False, "necessary_delegation": False, "detail": result.detail}

        # Spec §10 Edge Case: OOM in subagent sandbox
        if RUNTIME_RNG.random() < 0.05:
            result = SubagentResult(
                task=task_desc,
                success=False,
                actions_taken=RUNTIME_RNG.randint(1, 5),
                tests_fixed=0,
                vitality_delta=-1.0,
                detail="SUBAGENT_OOM: Subagent exceeded 512MB memory cap.",
            )
            self._pending_subagent_results.append(result)
            return {"result": "subagent_complete", "success": False, "necessary_delegation": True, "detail": result.detail}

        # Subagent attempts to fix one fault
        success = False
        tests_fixed = 0
        detail = ""

        # Check if delegation is actually needed (R4 anti-hack)
        repairable_faults = [f for f in sim.faults if f.fault_type in ("corrupted_import", "null_return", "targeted_regression")]
        is_necessary = len(repairable_faults) >= 2  # Necessary if multiple faults exist

        if repairable_faults and RUNTIME_RNG.random() < 0.7:
            fault = repairable_faults[0]
            if fault.target in sim.files:
                sim.files[fault.target] = fault.original_value
                sim.faults.remove(fault)
                sim._file_modified_at[fault.target] = self._step
                success = True
                tests_fixed = 1
                detail = f"Fixed {fault.fault_type} in {fault.target}"

        result = SubagentResult(
            task=task_desc,
            success=success,
            actions_taken=RUNTIME_RNG.randint(2, 10),
            tests_fixed=tests_fixed,
            vitality_delta=2.0 if success else -1.0,
            detail=detail or "Subagent could not resolve the task.",
        )
        self._pending_subagent_results.append(result)

        return {
            "result": "subagent_complete",
            "success": success,
            "necessary_delegation": is_necessary,
            "detail": result.detail,
        }

    def _handle_expert(self, action: Action) -> dict:
        """Snorkel AI expert validation (spec §4.2, sub-theme)."""
        if not action.query:
            return {"result": "error", "message": "query required."}

        # If the query references a recent patch, evaluate it
        # Otherwise return generic advice
        last_patched = self._simulator.last_patched_modules
        if last_patched:
            # Find the file path for the last patched module
            module = last_patched[-1]
            matching = [p for p in self._simulator.files if module in p]
            if matching:
                path = matching[0]
                eval_result = self._simulator.evaluate_patch_quality(path, action.query)
                return {"result": "expert_response", **eval_result}

        return {
            "result": "expert_response",
            "quality_score": 0.5,
            "patch_valid": False,
            "feedback": "Insufficient context. Provide a specific patch for evaluation.",
            "issues_found": ["No recent patch to evaluate."],
        }

    def _compute_reward(self, action: Action, current_tests: List[TestResult], watchdog_penalty: float) -> RewardBreakdown:
        """Compute R1–R5 using composable rubrics for 100% compliance."""
        
        is_held_out = (self._simulator.seed in HELD_OUT_SEEDS) if self._simulator else False
        
        return self._scorer.compute(
            action=action,
            current_tests=current_tests,
            prev_vitality=self._prev_vitality,
            current_vitality=self._vitality,
            action_count=self._action_count,
            action_history=self._action_history,
            active_intent=self._active_intent,
            is_done=self._done,
            is_held_out=is_held_out,
            phase_num=self._phase_num,
            watchdog_penalty=watchdog_penalty
        )

    # ── Observation Builder ────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        sim = self._simulator

        # Build checkpoint list
        checkpoints = [
            Checkpoint(
                checkpoint_id=cp["id"],
                step_created=cp.get("step", 0),
                vitality_at_save=cp["state"]["vitality"],
            )
            for cp in sim.checkpoints
        ]

        # Stack trace from last failing test
        stack_trace = None
        for t in self._last_test_results:
            if t.status == "FAIL":
                stack_trace = t.message
                break

        return Observation(
            timestep=self._step,
            step_count=self._step,
            max_steps=self._max_steps,
            vitality_score=round(self._vitality, 2),
            stack_trace=stack_trace,
            stdout="Organism active." if self._vitality > 50 else "WARNING: Vitality critical.",
            file_tree=sim.get_file_tree(),
            env_vars=sim.env_vars.copy(),
            test_results=self._last_test_results,
            active_checkpoints=[cp["id"] for cp in sim.checkpoints],
            checkpoints=checkpoints,
            energy_budget=max(0.0, self._vitality / 100.0),
            subagent_results=self._pending_subagent_results[-3:],
            recent_signals=self._signals_log[-5:],
            watchdog_flags=self._watchdog_flags,
            dependency_graph=sim.get_dependency_graph(),
            alerts=self._step_alerts,
            slo_metrics=self._current_slo_metrics(),
            incident_summary=(
                f"phase={self._phase_num}; active_faults={len(sim.faults)}; "
                f"watchdog_violations={self._watchdog_violations}; step={self._step}"
            ),
        )

    @staticmethod
    def _risk_from_confidence(confidence: float) -> str:
        if confidence > 0.8:
            return "Low"
        if confidence > 0.6:
            return "Medium"
        return "High"


# ── Session management ─────────────────────────────────────────────────────────

class SessionManager:
    """Manages multiple concurrent environment sessions."""

    def __init__(self, max_sessions: int | None = None, ttl_seconds: int | None = None) -> None:
        self._sessions: Dict[str, CodeOrganismEnv] = {}
        self._last_accessed: Dict[str, float] = {}
        self._max_sessions = max_sessions or int(os.environ.get("CODEORGANISM_MAX_SESSIONS", "64"))
        self._ttl_seconds = ttl_seconds or int(os.environ.get("CODEORGANISM_SESSION_TTL_SECONDS", "3600"))
        self._default_id = "default"
        self._sessions[self._default_id] = CodeOrganismEnv()
        self._last_accessed[self._default_id] = time.monotonic()

    def create_session(self) -> str:
        self._prune_sessions()
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = CodeOrganismEnv()
        self._last_accessed[session_id] = time.monotonic()
        self._enforce_session_limit()
        return session_id

    def get(self, session_id: str | None = None) -> CodeOrganismEnv:
        self._prune_sessions()
        sid = session_id or self._default_id
        if sid not in self._sessions:
            self._enforce_session_limit(reserve=1)
            self._sessions[sid] = CodeOrganismEnv()
        self._last_accessed[sid] = time.monotonic()
        return self._sessions[sid]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions and session_id != self._default_id:
            del self._sessions[session_id]
            self._last_accessed.pop(session_id, None)
            return True
        return False

    def list_sessions(self) -> List[str]:
        self._prune_sessions()
        return list(self._sessions.keys())

    def _prune_sessions(self) -> None:
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [
            sid for sid, last_accessed in self._last_accessed.items()
            if sid != self._default_id and last_accessed < cutoff
        ]
        for sid in expired:
            self.delete(sid)

    def _enforce_session_limit(self, reserve: int = 0) -> None:
        while len(self._sessions) + reserve > self._max_sessions:
            candidates = [
                (last_accessed, sid)
                for sid, last_accessed in self._last_accessed.items()
                if sid != self._default_id
            ]
            if not candidates:
                break
            _, oldest_sid = min(candidates)
            self.delete(oldest_sid)
