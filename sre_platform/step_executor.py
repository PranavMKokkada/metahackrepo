"""Single entry point for env steps/resets that keep the SRE platform layer in sync."""

from __future__ import annotations

from typing import Optional

from models import Action, Observation, StepResult
from sre_platform import services as sre_services
from sre_platform.routes import apply_production_or_guardrails_step
from sre_platform.state import STORE


def run_step_with_platform(env, action: Action, session_id: Optional[str] = None) -> StepResult:
    """Guardrails / production queue + env.step + post_step_enrich (same as HTTP /step)."""
    blocked_or_pending, _ = apply_production_or_guardrails_step(env, action, session_id)
    if blocked_or_pending is not None:
        return blocked_or_pending
    result = env.step(action)
    st = STORE.get(session_id)
    sre_services.post_step_enrich(env, action, result, st)
    return result


def reset_env_with_platform(env, task_id: str, session_id: Optional[str] = None, seed: Optional[int] = None) -> Observation:
    """env.reset + platform episode reset (same as HTTP /reset)."""
    obs = env.reset(task_id, seed=seed)
    st = STORE.get(session_id)
    sre_services.on_env_reset(st, env)
    return obs
