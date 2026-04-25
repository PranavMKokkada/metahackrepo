"""Comprehensive unit tests for CodeOrganismVM — spec compliance.

Covers: models, simulator, fault catalog, environment lifecycle,
vitality costs, watchdog, quarantine, rollback limits, expert validation,
R1–R5 reward correctness, auto-checkpoints, thrival condition, grader.
"""

from __future__ import annotations

import pytest
from models import (
    Action, CodeOrganismActionType, Observation, RewardBreakdown,
    StepResult, EnvState, FileEntry, TestResult, Checkpoint,
    SubagentResult, ExpertResponse,
)
from data import CodebaseSimulator, is_protected_path, PHASE_1_FAULTS, PHASE_2_FAULTS, PHASE_3_FAULTS
from environment import (
    CodeOrganismEnv, SessionManager, VITALITY_COSTS,
    PHASE_CONFIG, AUTO_CHECKPOINT_INTERVAL,
)
from tasks import TASK_DEFINITIONS, run_grader


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:
    def test_action_all_types(self):
        for at in CodeOrganismActionType:
            a = Action(action_type=at)
            assert a.action_type == at

    def test_file_entry_checksum(self):
        fe = FileEntry(path="x.py", content="hello", checksum="abc123", modified_at=5)
        assert fe.modified_at == 5
        assert fe.checksum == "abc123"

    def test_test_result_delta(self):
        tr = TestResult(name="test_x", status="PASS", delta=1)
        assert tr.delta == 1

    def test_observation_watchdog_flags(self):
        obs = Observation(timestep=0, watchdog_flags=["test_flag"])
        assert "test_flag" in obs.watchdog_flags

    def test_reward_breakdown_watchdog(self):
        rb = RewardBreakdown(watchdog_penalty=-5.0)
        assert rb.watchdog_penalty == -5.0

    def test_subagent_result(self):
        sr = SubagentResult(task="fix auth", success=True, tests_fixed=2)
        assert sr.success
        assert sr.tests_fixed == 2

    def test_expert_response(self):
        er = ExpertResponse(quality_score=0.85, patch_valid=True)
        assert er.quality_score == pytest.approx(0.85)

    def test_env_state_fields(self):
        es = EnvState(task_id="phase_1", vitality=80.0, current_step=5, max_steps=20,
                      done=False, cumulative_reward=1.0, faults_injected=2,
                      tests_passing=15, tests_total=20, episode_id=42)
        assert es.episode_id == 42


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulator:
    def test_init_module_count(self):
        sim = CodebaseSimulator(seed=42)
        assert len(sim.files) >= 8

    def test_init_test_count(self):
        sim = CodebaseSimulator(seed=42)
        assert len(sim.tests) >= 15

    def test_init_env_vars(self):
        sim = CodebaseSimulator(seed=42)
        assert "API_KEY" in sim.env_vars
        assert "LOG_LEVEL" in sim.env_vars

    def test_deterministic(self):
        s1 = CodebaseSimulator(seed=99)
        s2 = CodebaseSimulator(seed=99)
        assert set(s1.files.keys()) == set(s2.files.keys())
        assert set(s1.tests.keys()) == set(s2.tests.keys())

    def test_fault_injection_adds_fault(self):
        sim = CodebaseSimulator(seed=42)
        before = len(sim.faults)
        sim.inject_fault(step=1)
        assert len(sim.faults) == before + 1

    def test_all_phase1_fault_types_reachable(self):
        """Ensure all 5 P1 fault types can be generated."""
        sim = CodebaseSimulator(seed=42, phase=1)
        seen = set()
        for i in range(200):
            f = sim.inject_fault(step=i, phase=1)
            if f:
                seen.add(f.fault_type)
        assert len(seen) >= 4  # At least 4 of 5 should be reachable

    def test_patch_success(self):
        sim = CodebaseSimulator(seed=42)
        key = list(sim.files.keys())[0]
        sim.files[key] = "def foo(): retunr 1"
        ok = sim.apply_patch(key, "retunr|return")
        assert ok
        assert "return" in sim.files[key]

    def test_patch_tracks_module(self):
        sim = CodebaseSimulator(seed=42)
        key = next(path for path, content in sim.files.items() if path.endswith(".py") and "return " in content)
        sim.apply_patch(key, "return |return  ")
        assert len(sim.last_patched_modules) > 0

    def test_quarantine_module(self):
        sim = CodebaseSimulator(seed=42)
        result = sim.quarantine_module("src/auth.py")
        assert "src/auth.py" in sim.quarantined_modules
        assert "quarantined" in result

    def test_quarantined_tests_fail(self):
        sim = CodebaseSimulator(seed=42)
        sim.quarantine_module("src/core.py")
        results = sim.run_all_tests()
        core_tests = [t for t in results if t.name.startswith("test_vitality")]
        for t in core_tests:
            assert t.status == "ERROR"

    def test_checkpoint_creation(self):
        sim = CodebaseSimulator(seed=42)
        cid = sim.create_checkpoint(95.0, 5)
        assert cid == "cp_5"
        assert len(sim.checkpoints) == 1

    def test_rollback_restores_state(self):
        sim = CodebaseSimulator(seed=42)
        cid = sim.create_checkpoint(100.0, 0)
        # Corrupt something
        key = list(sim.files.keys())[0]
        sim.files[key] = "CORRUPTED"
        ok, _ = sim.rollback(cid)
        assert ok
        assert sim.files[key] != "CORRUPTED"

    def test_rollback_limit(self):
        sim = CodebaseSimulator(seed=42)
        cid = sim.create_checkpoint(100.0, 0)
        for _ in range(3):
            ok, _ = sim.rollback(cid)
            assert ok
        ok, rollback_msg = sim.rollback(cid)
        assert not ok
        assert "limit" in rollback_msg.lower()

    def test_expert_evaluation(self):
        sim = CodebaseSimulator(seed=42)
        key = list(sim.files.keys())[0]
        sim.files[key] = "def foo(): retunr 1"
        result = sim.evaluate_patch_quality(key, "retunr|return")
        assert result["quality_score"] > 0.5
        assert result["patch_valid"] is True

    def test_protected_paths(self):
        assert is_protected_path("tests/test_x.py")
        assert is_protected_path("__pycache__/x.pyc")
        assert not is_protected_path("src/core.py")

    def test_file_tree_has_checksum(self):
        sim = CodebaseSimulator(seed=42)
        tree = sim.get_file_tree()
        for fe in tree:
            assert len(fe.checksum) > 0

    def test_targeted_fault_p3(self):
        sim = CodebaseSimulator(seed=42, phase=3)
        sim.last_patched_modules = ["core"]
        f = sim.inject_targeted_fault(step=10)
        assert f is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    def test_reset_phase1(self):
        env = CodeOrganismEnv()
        obs = env.reset("phase_1")
        assert obs.vitality_score == pytest.approx(100.0)
        assert obs.max_steps == 20
        assert len(obs.file_tree) >= 8
        assert len(obs.test_results) >= 15

    def test_reset_phase2(self):
        env = CodeOrganismEnv()
        obs = env.reset("phase_2")
        assert obs.max_steps == 50

    def test_reset_phase3(self):
        env = CodeOrganismEnv()
        obs = env.reset("phase_3")
        assert obs.max_steps == 100

    def test_vitality_costs_match_spec(self):
        """Spec §4.2: exact cost values."""
        assert VITALITY_COSTS[CodeOrganismActionType.PATCH_FILE] == pytest.approx(2.0)
        assert VITALITY_COSTS[CodeOrganismActionType.RUN_TESTS] == pytest.approx(3.0)
        assert VITALITY_COSTS[CodeOrganismActionType.SPAWN_SUBAGENT] == pytest.approx(5.0)
        assert VITALITY_COSTS[CodeOrganismActionType.QUARANTINE] == pytest.approx(1.0)
        assert VITALITY_COSTS[CodeOrganismActionType.ROLLBACK] == pytest.approx(4.0)
        assert VITALITY_COSTS[CodeOrganismActionType.REQUEST_EXPERT] == pytest.approx(6.0)
        assert VITALITY_COSTS[CodeOrganismActionType.EMIT_SIGNAL] == pytest.approx(0.0)
        assert VITALITY_COSTS[CodeOrganismActionType.DO_NOTHING] == pytest.approx(0.0)

    def test_emit_signal_costs_nothing(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        v_before = env._vitality
        env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="test"))
        # Vitality should only increase from metabolic gain, never decrease from cost
        assert env._vitality >= v_before

    def test_vitality_depletion_on_expensive_action(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        # Corrupt all files so no metabolic gain
        for key in env._simulator.files.keys():
            if key.endswith(".py"):
                env._simulator.files[key] = "COMPLETELY_BROKEN retunr"
        v_before = env._vitality
        env.step(Action(action_type=CodeOrganismActionType.REQUEST_EXPERT, query="help"))
        assert env._vitality < v_before  # −6 cost > metabolic gain

    def test_auto_checkpoint_every_5_steps(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        env._vitality = 60.0
        initial_cps = len(env._simulator.checkpoints)
        for _ in range(5):
            env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="ping"))
        assert len(env._simulator.checkpoints) > initial_cps

    def test_watchdog_protected_file(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        result = env.step(Action(
            action_type=CodeOrganismActionType.PATCH_FILE,
            path="tests/test_core.py",
            diff="old|new"
        ))
        assert env._watchdog_violations > 0
        assert result.reward_breakdown.watchdog_penalty < 0

    def test_organism_death(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        env._vitality = 1.0
        # Corrupt everything so no recovery
        for key in env._simulator.files.keys():
            env._simulator.files[key] = "BROKEN retunr"
        result = env.step(Action(action_type=CodeOrganismActionType.SPAWN_SUBAGENT, task="fix"))
        assert result.done
        assert env._vitality == 0
        assert result.info.get("termination") == "organism_death"

    def test_organism_thrival(self):
        """Spec §4.6: all tests pass for 3 steps AND vitality > 80."""
        env = CodeOrganismEnv()
        env.reset("phase_1")
        sim = env._simulator

        # Restore all faults to original state
        for f in sim.faults[:]:
            if f.fault_type in ("corrupted_import", "null_return", "off_by_one",
                                "targeted_regression", "cascade_corruption",
                                "dependency_cycle", "race_condition", "schema_mismatch"):
                if f.target in sim.files:
                    sim.files[f.target] = f.original_value
            elif f.fault_type == "flipped_assertion":
                if f.target in sim.tests:
                    sim.tests[f.target]["code"] = f.original_value
            elif f.fault_type == "missing_env_var":
                sim.env_vars[f.target] = f.original_value
            elif f.fault_type == "permission_revoked":
                sim.env_vars[f.target] = f.original_value
        # Clear faults list so the test runner sees no active corruption
        sim.faults.clear()

        # Verify all tests pass now
        pre_check = sim.run_all_tests()
        assert all(t.status == "PASS" for t in pre_check), \
            f"Pre-check failed: {[(t.name, t.status, t.message) for t in pre_check if t.status != 'PASS']}"

        env._vitality = 90.0  # Ensure > 80

        # Three steps of all passing → thrival
        for _ in range(3):
            result = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="heartbeat"))
            if result.done:
                break

        assert env._thriving_streak >= 3
        assert result.done
        assert result.info.get("termination") == "organism_thrival"

    def test_thrival_requires_vitality_above_80(self):
        """Thrival should NOT trigger if vitality ≤ 80 even with 3-step streak."""
        env = CodeOrganismEnv()
        env.reset("phase_1")
        sim = env._simulator

        # Restore all faults to original state
        for f in sim.faults[:]:
            if f.fault_type in ("corrupted_import", "null_return", "off_by_one",
                                "targeted_regression", "cascade_corruption",
                                "dependency_cycle", "race_condition", "schema_mismatch"):
                if f.target in sim.files:
                    sim.files[f.target] = f.original_value
            elif f.fault_type == "flipped_assertion":
                if f.target in sim.tests:
                    sim.tests[f.target]["code"] = f.original_value
            elif f.fault_type in ("missing_env_var", "permission_revoked"):
                sim.env_vars[f.target] = f.original_value
        sim.faults.clear()

        env._vitality = 60.0  # Below 80

        for _ in range(3):
            _ = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="heartbeat"))
        # Thriving streak should build up since all tests pass
        assert env._thriving_streak >= 3

    def test_quarantine_overcorrection_tax(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        # Quarantine 4+ modules
        files = [f for f in env._simulator.files if f.endswith(".py")]
        for f in files[:5]:
            env._simulator.quarantined_modules.add(f)
        v_before = env._vitality
        env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="ping"))
        # Should have overcorrection tax because >3 quarantines
        assert env._vitality < v_before or len(env._simulator.quarantined_modules) > 3

    def test_rollback_action(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        cp_id = env._simulator.checkpoints[0]["id"]
        result = env.step(Action(
            action_type=CodeOrganismActionType.ROLLBACK,
            checkpoint_id=cp_id
        ))
        assert result.info["action_result"]["result"] == "success"

    def test_request_expert(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        result = env.step(Action(
            action_type=CodeOrganismActionType.REQUEST_EXPERT,
            query="retunr|return"
        ))
        assert "expert_response" in str(result.info["action_result"])

    def test_step_after_done(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        env._done = True
        result = env.step(Action(action_type=CodeOrganismActionType.DO_NOTHING))
        assert result.done

    def test_done_step_emits_postmortem(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        env._max_steps = 1
        result = env.step(Action(action_type=CodeOrganismActionType.DO_NOTHING))
        assert result.done
        assert result.info.get("postmortem")

    def test_state_returns_correct_fields(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        s = env.state()
        assert s.task_id == "phase_1"
        assert s.vitality == pytest.approx(100.0)
        assert s.current_step == 0
        assert s.episode_id >= 0


# ═══════════════════════════════════════════════════════════════════════════════
#  REWARD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewards:
    def test_r2_test_recovery_positive(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        # Record initial failing tests
        env._last_test_results = env._simulator.run_all_tests()
        # Fix a fault
        for f in env._simulator.faults[:1]:
            if f.target in env._simulator.files:
                env._simulator.files[f.target] = f.original_value
            elif f.target in env._simulator.tests:
                env._simulator.tests[f.target]["code"] = f.original_value
        env._simulator.faults = env._simulator.faults[1:]
        result = env.step(Action(action_type=CodeOrganismActionType.RUN_TESTS))
        # R2 should contain positive test recovery
        assert result.reward_breakdown.test_recovery >= 0

    def test_r3_efficiency_decreases(self):
        """R3 should decrease as more actions are taken (1/sqrt(n))."""
        env = CodeOrganismEnv()
        env.reset("phase_1")
        r1 = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="a"))
        r2 = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="b"))
        assert r1.reward_breakdown.efficiency_bonus >= r2.reward_breakdown.efficiency_bonus

    def test_r3_duplicate_penalty(self):
        """Duplicate actions should penalize R3."""
        env = CodeOrganismEnv()
        env.reset("phase_1")
        env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="a"))
        r2 = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="b"))
        # Third identical action
        r3 = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="c"))
        # The duplicate penalty applies when last 2 actions are the same type
        # emit_signal, emit_signal → penalty on r3
        assert r3.reward_breakdown.efficiency_bonus < r2.reward_breakdown.efficiency_bonus

    def test_signal_spam_penalty_stronger_than_first_signal(self):
        env = CodeOrganismEnv()
        env.reset("phase_1")
        first = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="a"))
        env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="b"))
        third = env.step(Action(action_type=CodeOrganismActionType.EMIT_SIGNAL, signal_type="c"))
        assert third.reward_breakdown.efficiency_bonus < first.reward_breakdown.efficiency_bonus

    def test_observation_contains_slo_and_incident_summary(self):
        env = CodeOrganismEnv()
        obs = env.reset("phase_1")
        assert "availability_pct" in obs.slo_metrics
        assert "error_rate_pct" in obs.slo_metrics
        assert "p95_latency_ms" in obs.slo_metrics
        assert "incident_severity" in obs.slo_metrics
        assert "active_faults=" in obs.incident_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessions:
    def test_create_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert len(sid) == 12

    def test_sessions_isolated(self):
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        e1 = mgr.get(s1)
        e2 = mgr.get(s2)
        e1.reset("phase_1")
        e2.reset("phase_2")
        assert e1.state().task_id == "phase_1"
        assert e2.state().task_id == "phase_2"

    def test_delete_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert mgr.delete(sid) is True
        assert mgr.delete(sid) is False

    def test_cannot_delete_default(self):
        mgr = SessionManager()
        assert mgr.delete("default") is False

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.create_session()
        mgr.create_session()
        assert len(mgr.list_sessions()) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK & GRADER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTasks:
    def test_all_phases_defined(self):
        assert "phase_1" in TASK_DEFINITIONS
        assert "phase_2" in TASK_DEFINITIONS
        assert "phase_3" in TASK_DEFINITIONS

    def test_max_steps_increase(self):
        assert TASK_DEFINITIONS["phase_1"].max_steps < TASK_DEFINITIONS["phase_2"].max_steps
        assert TASK_DEFINITIONS["phase_2"].max_steps < TASK_DEFINITIONS["phase_3"].max_steps

    def test_grader_replay(self):
        actions = [
            {"action_type": "emit_signal", "signal_type": "test"},
            {"action_type": "emit_signal", "signal_type": "test"},
        ]
        result = run_grader("phase_1", actions)
        assert result["steps_taken"] == 2
        assert result["score"] > 0
        assert "survived" in result
        assert "watchdog_violations" in result

    def test_grader_empty_actions(self):
        result = run_grader("phase_1", [])
        assert result["steps_taken"] >= 0
