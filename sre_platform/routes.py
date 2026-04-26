"""HTTP API for the enterprise SRE platform layer."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from environment import CodeOrganismEnv
from models import Action, CodeOrganismActionType, RewardBreakdown, StepResult

from sre_platform import services
from sre_platform.state import STORE, PendingProductionBatch


def _pending_to_dict(p: Optional[PendingProductionBatch]) -> Optional[Dict[str, Any]]:
    if not p:
        return None
    return {
        "batch_id": p.batch_id,
        "suggestions": [
            {
                "suggestion_id": s.suggestion_id,
                "rank": s.rank,
                "score": s.score,
                "path": s.path,
                "diff": s.diff,
                "rationale": s.rationale,
            }
            for s in p.suggestions
        ],
    }


def _cicd_to_dict(run) -> Optional[Dict[str, Any]]:
    if not run:
        return None
    return {
        "run_id": run.run_id,
        "episode_id": run.episode_id,
        "stages": [{"name": s.name, "status": s.status, "detail": s.detail} for s in run.stages],
        "current_stage_index": run.current_stage_index,
    }


def _mem_to_dict(m) -> Dict[str, Any]:
    return {
        "entry_id": m.entry_id,
        "fault_signature": m.fault_signature,
        "strategy": m.strategy,
        "outcome": m.outcome,
        "task_id": m.task_id,
    }


def _pred_to_dict(p) -> Dict[str, Any]:
    return {"id": p.alert_id, "severity": p.severity, "pattern": p.pattern, "recommendation": p.recommendation}


def build_platform_router(
    get_env: Callable[[Optional[str]], CodeOrganismEnv],
    require_api_key,
) -> APIRouter:
    router = APIRouter(prefix="/platform", tags=["platform"])

    class ToggleBody(BaseModel):
        enabled: bool

    class GuardrailBody(BaseModel):
        rollback_confidence_min: Optional[float] = None
        restricted_paths_extra: Optional[List[str]] = None
        safe_zones: Optional[List[str]] = None
        catastrophic_block: Optional[bool] = None

    class ApproveBody(BaseModel):
        suggestion_id: str

    class LogIngestBody(BaseModel):
        lines: List[str] = Field(default_factory=list)
        source: str = "external"

    @router.get("/session/state")
    def platform_snapshot(
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        env = get_env(x_session_id)
        preds = services.predictive_scan(env)
        return {
            "production_mode": st.production_mode,
            "pending_suggestion_batch": _pending_to_dict(st.pending),
            "guardrails": {
                "rollback_confidence_min": st.rollback_confidence_min,
                "restricted_paths_extra": st.restricted_paths_extra,
                "safe_zones": st.safe_zones,
                "catastrophic_block": st.catastrophic_block,
            },
            "cicd": {
                "active": _cicd_to_dict(st.active_pipeline),
                "history": [_cicd_to_dict(r) for r in st.cicd_runs[-8:]],
            },
            "business": st.business,
            "memory_recent": [_mem_to_dict(m) for m in st.memory_log[-12:]],
            "evolution": services.evolution_series(st),
            "predictive": [_pred_to_dict(p) for p in preds],
            "ingested_logs_tail": st.ingested_logs[-40:],
            "episode": {
                "task_id": env._task_id,
                "step": env._step,
                "vitality": env._vitality,
                "done": env._done,
            },
        }

    @router.post("/session/production-mode")
    def set_production_mode(
        body: ToggleBody,
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        st.production_mode = body.enabled
        return {"production_mode": st.production_mode}

    @router.post("/session/guardrails")
    def set_guardrails(
        body: GuardrailBody,
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        if body.rollback_confidence_min is not None:
            st.rollback_confidence_min = max(0.0, min(1.0, body.rollback_confidence_min))
        if body.restricted_paths_extra is not None:
            st.restricted_paths_extra = body.restricted_paths_extra
        if body.safe_zones is not None:
            st.safe_zones = body.safe_zones
        if body.catastrophic_block is not None:
            st.catastrophic_block = body.catastrophic_block
        return {"ok": True}

    @router.post("/session/production/approve")
    def approve_suggestion(
        body: ApproveBody,
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        env = get_env(x_session_id)
        if not st.pending:
            raise HTTPException(400, "No pending suggestions to approve.")
        chosen = next((s for s in st.pending.suggestions if s.suggestion_id == body.suggestion_id), None)
        if not chosen:
            raise HTTPException(404, "Unknown suggestion_id")
        action = Action(
            action_type=CodeOrganismActionType.PATCH_FILE,
            path=chosen.path,
            diff=chosen.diff,
            justification=f"Human-approved suggestion {chosen.suggestion_id} (production mode).",
        )
        err = services.validate_guardrails(env, action, st)
        if err:
            raise HTTPException(403, err)
        result = env.step(action)
        st.pending = None
        services.post_step_enrich(env, action, result, st)
        return result

    @router.post("/session/production/reject")
    def reject_pending(
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        st.pending = None
        return {"rejected": True}

    @router.post("/session/logs/ingest")
    def ingest_logs(
        body: LogIngestBody,
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        prefix = f"[{body.source}] "
        for line in body.lines[-500:]:
            st.ingested_logs.append(prefix + line)
        if len(st.ingested_logs) > services.MAX_INGESTED_LOG_LINES:
            st.ingested_logs = st.ingested_logs[-services.MAX_INGESTED_LOG_LINES :]
        return {"ingested": len(body.lines)}

    @router.get("/session/memory/lookup")
    def memory_lookup(
        x_session_id: Optional[str] = Header(None),
        _auth: None = Depends(require_api_key),
    ):
        st = STORE.get(x_session_id)
        env = get_env(x_session_id)
        sig = services.fault_signature_from_env(env)
        return {"fault_signature": sig, "matches": services.memory_insights(st, sig)}

    return router


def apply_production_or_guardrails_step(
    env: CodeOrganismEnv,
    action: Action,
    session_id: Optional[str],
) -> tuple[Optional[StepResult], Optional[str]]:
    """Returns (StepResult, None) if handled without calling env.step; (None, None) if caller should env.step.

    If (None, error_message) guardrail blocked before production branch.
    """
    st = STORE.get(session_id)
    err = services.validate_guardrails(env, action, st)
    if err:
        st.touch()
        obs = env._make_observation() if env._simulator else None
        return (
            StepResult(
                observation=obs,
                reward=-0.5,
                reward_breakdown=RewardBreakdown(total=-0.5, watchdog_penalty=-0.5),
                done=False,
                info={
                    "error": err,
                    "guardrail_block": True,
                    "explanation": {
                        "root_cause": "Policy guardrail",
                        "reasoning": err,
                        "confidence_pct": 0.0,
                        "impact": "No environment mutation applied.",
                        "fix_summary": "blocked",
                    },
                },
            ),
            None,
        )

    if st.production_mode and action.action_type == CodeOrganismActionType.PATCH_FILE:
        sug = services.build_patch_suggestions(env, action)
        import time as _time
        import uuid as _uuid

        st.pending = PendingProductionBatch(
            batch_id=_uuid.uuid4().hex[:12],
            created_at=_time.time(),
            suggestions=sug,
            original_justification=action.justification or "",
        )
        st.touch()
        obs = env._make_observation() if env._simulator else None
        preds = services.predictive_scan(env)
        return (
            StepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={
                    "production_mode": True,
                    "pending_human_review": True,
                    "predictive_alerts": [
                        {"id": p.alert_id, "severity": p.severity, "pattern": p.pattern, "recommendation": p.recommendation}
                        for p in preds
                    ],
                    "suggestions": [
                        {
                            "suggestion_id": s.suggestion_id,
                            "rank": s.rank,
                            "score": s.score,
                            "path": s.path,
                            "diff": s.diff,
                            "rationale": s.rationale,
                        }
                        for s in sug
                    ],
                    "explanation": {
                        "root_cause": "Awaiting human approval — patch not applied.",
                        "reasoning": "Production mode routes PATCH_FILE through ranked suggestions.",
                        "confidence_pct": 0.0,
                        "impact": "No filesystem mutation until POST /platform/session/production/approve.",
                        "fix_summary": "none",
                    },
                },
            ),
            None,
        )
    return (None, None)
