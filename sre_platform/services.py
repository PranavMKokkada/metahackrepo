"""Core logic for the 10 enterprise SRE platform capabilities."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from data import is_protected_path
from models import Action, CodeOrganismActionType, StepResult

from sre_platform.state import (
    CICDPipelineRun,
    CICDStage,
    EvolutionSnapshot,
    MemoryEntry,
    PatchSuggestion,
    PredictiveAlert,
    SessionPlatformState,
)


def fault_signature_from_env(env) -> str:
    sim = env._simulator
    if not sim or not sim.faults:
        return "no_active_faults"
    parts = sorted(f"{f.fault_type}:{f.target}" for f in sim.faults[:8])
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return h


def build_patch_suggestions(env, action: Action) -> List[PatchSuggestion]:
    """Rank candidate patches from live faults, failing tests → file map, and operator diff."""
    sim = env._simulator
    obs = env._make_observation() if sim else None
    candidates: List[Tuple[str, str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    def add(path: str, diff: str, rationale: str, score: float) -> None:
        key = (path, diff)
        if key in seen or not path:
            return
        seen.add(key)
        candidates.append((path, diff, rationale, score))

    if action.path and action.diff:
        add(str(action.path), str(action.diff), "Operator-supplied diff (primary)", 0.95)

    if sim and obs:
        # Active faults whose target is a source file in the simulator.
        for f in sim.faults[:16]:
            tgt = getattr(f, "target", "") or ""
            if not isinstance(tgt, str) or not tgt.endswith(".py") or tgt not in sim.files:
                continue
            body = sim.files[tgt]
            ft = getattr(f, "fault_type", "")
            if ft == "corrupted_import" and ("improt" in body or "nonexistent_module" in body):
                add(tgt, "improt |import ", f"Repair import corruption ({f.fault_id})", 0.88)

        # Failing tests → owning file from the same test catalog the simulator uses.
        for t in obs.test_results:
            if t.status == "PASS":
                continue
            meta = sim.tests.get(t.name) or {}
            path = meta.get("file")
            if not path or path not in sim.files:
                continue
            content = sim.files[path]
            msg = (t.message or "").lower()
            if "import" in msg or "improt" in content:
                add(path, "improt |import ", f"Import stability from failing `{t.name}`", 0.74)
            if "retunr" in content:
                add(path, "retunr|return", f"Syntax repair from failing `{t.name}`", 0.72)
            if "deaf " in content:
                add(path, "deaf |def ", f"Keyword repair from failing `{t.name}`", 0.71)
            break

    if len(candidates) == 1 and action.path and action.diff:
        path = str(action.path)
        alt = action.diff.replace("|", "|#alt:") if "|" in str(action.diff) else f"{action.diff}|noop"
        add(path, alt, "Conservative variant (lower blast radius)", 0.62)

    if not candidates and sim and sim.files:
        for path in sorted(sim.files):
            if not path.endswith(".py"):
                continue
            c = sim.files[path]
            for typo, fix in (("retunr", "return"), ("improt ", "import "), ("deaf ", "def ")):
                if typo in c:
                    add(path, f"{typo}|{fix}", f"Live file scan: `{typo}` in `{path}`", 0.52)
                    break
            if candidates:
                break

    out: List[PatchSuggestion] = []
    for i, (path, diff, rationale, score) in enumerate(sorted(candidates, key=lambda x: -x[3])):
        out.append(
            PatchSuggestion(
                suggestion_id=uuid.uuid4().hex[:10],
                rank=i + 1,
                score=round(score, 3),
                path=path,
                diff=diff,
                rationale=rationale,
            )
        )
    return out[:5]


MAX_INGESTED_LOG_LINES = 4000


def post_step_enrich(env, action: Action, result: StepResult, st: SessionPlatformState) -> None:
    """Shared post-step hooks for HTTP /step, MCP, and production approve paths."""
    st.touch()
    record_evolution(st, env, action)
    advance_cicd_on_recovery(st, result.reward_breakdown.test_recovery > 0)
    result.info["explanation"] = explain_step(env, action, result)
    preds = predictive_scan(env)
    st.last_predictions = preds
    result.info["predictive_alerts"] = [
        {"id": p.alert_id, "severity": p.severity, "pattern": p.pattern, "recommendation": p.recommendation}
        for p in preds
    ]
    if action.action_type == CodeOrganismActionType.SPAWN_SUBAGENT:
        result.info["specialized_agent"] = specialized_subagent_detail(action.task)
    if st.ingested_logs:
        result.info["external_log_tail"] = st.ingested_logs[-12:]
    update_business_metrics(st, env, result)
    if result.done:
        record_memory(
            st,
            env,
            outcome=str(result.info.get("termination", "done")),
            strategy=action.action_type.value,
        )


def explain_step(env, action: Action, result: StepResult) -> Dict[str, Any]:
    """Structured explainability from environment + step outcome."""
    sim = env._simulator
    root = "No active fault catalog match."
    if sim and sim.faults:
        f0 = sim.faults[0]
        root = f"{f0.fault_type} affecting {f0.target}"
    rb = result.reward_breakdown
    impact_tests = 0
    if result.observation:
        impact_tests = sum(1 for t in result.observation.test_results if getattr(t, "delta", 0) == 1)
    conf = float(result.info.get("sre_metrics", {}).get("confidence", 0.75))
    reasoning_parts = [
        f"Action {action.action_type.value} with justification length {len(action.justification or '')}.",
        f"Reward total {rb.total:.3f} (vitality_delta={rb.vitality_delta:.2f}, test_recovery={rb.test_recovery:.2f}).",
    ]
    if getattr(env, "_watchdog_flags", None):
        reasoning_parts.append(f"Watchdog: {env._watchdog_flags}")
    return {
        "root_cause": root,
        "reasoning": " ".join(reasoning_parts),
        "confidence_pct": round(min(0.99, max(0.05, conf)) * 100, 1),
        "impact": f"Recovered or stabilized {impact_tests} test(s) this step; cumulative reward {env.state().cumulative_reward:.3f}.",
        "fix_summary": result.info.get("action_result", {}).get("result") if isinstance(result.info.get("action_result"), dict) else str(result.info.get("action_result", "")),
    }


def validate_guardrails(env, action: Action, st: SessionPlatformState) -> Optional[str]:
    """Return error message if blocked, else None."""
    if action.action_type == CodeOrganismActionType.PATCH_FILE and action.path:
        if is_protected_path(action.path):
            return f"Guardrail: path '{action.path}' is protected (read-only / system zone)."
        for pat in st.restricted_paths_extra:
            if pat and pat in action.path.replace("\\", "/"):
                return f"Guardrail: path matches restricted pattern '{pat}'."
        pl = action.path.lower()
        for zone in st.safe_zones:
            if zone and zone in pl:
                return f"Guardrail: '{action.path}' is in AI safe-zone '{zone}' (no direct patch)."
    if action.action_type == CodeOrganismActionType.ROLLBACK:
        conf = float(getattr(env, "_last_action_confidence", 0.0) or 0.0)
        if conf <= 0.0:
            conf = float(st.business.get("last_confidence", 0.85))
        if conf < st.rollback_confidence_min:
            return (
                f"Guardrail: rollback blocked — confidence {conf:.2f} below threshold "
                f"{st.rollback_confidence_min:.2f} (catastrophic failure prevention)."
            )
    if st.catastrophic_block and action.action_type == CodeOrganismActionType.QUARANTINE:
        if env._simulator and len(env._simulator.quarantined_modules) >= 5:
            return "Guardrail: quarantine blocked — blast-radius cap reached."
    return None


def specialized_subagent_detail(task: Optional[str]) -> Dict[str, Any]:
    t = (task or "").lower()
    if "debug" in t or "diagnos" in t:
        return {"agent": "DebugAgent", "focus": "trace_correlation", "artifact": "thread_dump_summary.json"}
    if "patch" in t or "fix" in t or "hotfix" in t:
        return {"agent": "PatchAgent", "focus": "minimal_diff", "artifact": "patch_plan.md"}
    if "test" in t or "verif" in t:
        return {"agent": "TestAgent", "focus": "coverage_delta", "artifact": "junit_aggregate.xml"}
    return {"agent": "CoordinatorAgent", "focus": "general", "artifact": "handoff_notes.txt"}


def predictive_scan(env) -> List[PredictiveAlert]:
    """Heuristic proactive signals from current observation (no separate model)."""
    if env._done or not env._simulator:
        return []
    obs = env._make_observation()
    alerts: List[PredictiveAlert] = []
    fail_ratio = sum(1 for t in obs.test_results if t.status != "PASS") / max(1, len(obs.test_results))
    if fail_ratio > 0.25 and obs.vitality_score > 70:
        alerts.append(
            PredictiveAlert(
                alert_id=uuid.uuid4().hex[:8],
                severity="warn",
                pattern="rising_test_entropy",
                recommendation="Run diagnostics before next fault window; pre-stage rollback checkpoint.",
                ts=time.time(),
            )
        )
    if env._vitality < env._prev_vitality - 8:
        alerts.append(
            PredictiveAlert(
                alert_id=uuid.uuid4().hex[:8],
                severity="high",
                pattern="vitality_cliff",
                recommendation="Predictive stabilization: emit INTENT_PATCH and narrow blast radius.",
                ts=time.time(),
            )
        )
    if len(obs.stack_trace or "") > 200 and fail_ratio > 0:
        alerts.append(
            PredictiveAlert(
                alert_id=uuid.uuid4().hex[:8],
                severity="warn",
                pattern="stack_trace_growth",
                recommendation="Pre-failure window: isolate last touched module via quarantine dry-run.",
                ts=time.time(),
            )
        )
    return alerts


def on_env_reset(st: SessionPlatformState, env) -> None:
    """New episode: clear pending/evolution timeline; start a fresh CI/CD run."""
    st.pending = None
    st.evolution.clear()
    st.step_timestamps.clear()
    start_cicd_pipeline(st, int(getattr(env, "_episode_id", 0)))


def start_cicd_pipeline(st: SessionPlatformState, episode_id: int) -> CICDPipelineRun:
    now = time.time()
    stages = [
        CICDStage("commit", "failed", "Synthetic main drift detected", now),
        CICDStage("pull_request", "pending", "Awaiting remediation", now),
        CICDStage("ci_build", "pending", "Queued", now),
        CICDStage("ci_test", "pending", "Queued", now),
        CICDStage("deploy_preview", "pending", "Queued", now),
    ]
    run = CICDPipelineRun(run_id=uuid.uuid4().hex[:12], episode_id=episode_id, stages=stages, current_stage_index=0)
    st.cicd_runs.append(run)
    st.active_pipeline = run
    return run


def advance_cicd_on_recovery(st: SessionPlatformState, had_test_recovery: bool) -> None:
    run = st.active_pipeline
    if not run:
        return
    if had_test_recovery and run.current_stage_index < len(run.stages):
        idx = run.current_stage_index
        run.stages[idx].status = "success"
        run.stages[idx].detail = "Recovered via autonomous remediation"
        if idx + 1 < len(run.stages):
            run.current_stage_index = idx + 1
            run.stages[idx + 1].status = "running"
            run.stages[idx + 1].detail = "PR checks re-triggered"
        else:
            run.stages[-1].status = "success"
            run.stages[-1].detail = "Pipeline green"


def record_memory(st: SessionPlatformState, env, outcome: str, strategy: str) -> None:
    st.memory_log.append(
        MemoryEntry(
            entry_id=uuid.uuid4().hex[:10],
            ts=time.time(),
            fault_signature=fault_signature_from_env(env),
            strategy=strategy,
            outcome=outcome,
            task_id=env._task_id,
        )
    )


def record_evolution(st: SessionPlatformState, env, action: Action) -> None:
    st.evolution.append(
        EvolutionSnapshot(
            ts=time.time(),
            step=env._step,
            action_type=action.action_type.value,
            cumulative_reward=env.state().cumulative_reward,
            vitality=env._vitality,
        )
    )


def update_business_metrics(st: SessionPlatformState, env, result: StepResult) -> None:
    sre = result.info.get("sre_metrics", {})
    st.business["downtime_saved_seconds"] = float(sre.get("downtime_saved_total", st.business.get("downtime_saved_seconds", 0)))
    st.business["last_confidence"] = float(sre.get("confidence", 0.75))
    if result.done and (result.reward > 0 or (result.info.get("termination") == "organism_thrival")):
        st.incidents_auto_resolved += 1
    if result.done:
        st.incidents_total += 1
        if len(st.step_timestamps) >= 2:
            span = st.step_timestamps[-1] - st.step_timestamps[0]
            st.mttr_with_ai_seconds = max(30.0, span)
            st.business["mttr_with_ai_min"] = round(st.mttr_with_ai_seconds / 60.0, 2)
    st.business["mttr_baseline_min"] = round(st.mttr_baseline_seconds / 60.0, 2)
    denom = max(1, st.incidents_total)
    st.business["incidents_auto_resolved_pct"] = round(100.0 * st.incidents_auto_resolved / denom, 1)


def evolution_series(st: SessionPlatformState) -> Dict[str, Any]:
    """Aggregate for charts: action distribution over early vs late window."""
    if not st.evolution:
        return {"early": {}, "late": {}, "points": []}
    mid = len(st.evolution) // 2 or 1
    early = st.evolution[:mid]
    late = st.evolution[mid:]
    def bucket(snaps: List[EvolutionSnapshot]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for s in snaps:
            d[s.action_type] = d.get(s.action_type, 0) + 1
        return d
    return {
        "early": bucket(early),
        "late": bucket(late),
        "points": [
            {"step": e.step, "action": e.action_type, "reward": e.cumulative_reward, "vitality": e.vitality}
            for e in st.evolution[-80:]
        ],
    }


def memory_insights(st: SessionPlatformState, signature: str) -> List[Dict[str, Any]]:
    return [
        {"fault_signature": m.fault_signature, "strategy": m.strategy, "outcome": m.outcome, "task_id": m.task_id}
        for m in st.memory_log
        if m.fault_signature == signature
    ][-5:]
