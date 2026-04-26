"""FastAPI application for Autonomous SRE OpenEnv environment."""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from typing import Optional, List, Deque, Dict

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import Action, CodeOrganismActionType, Observation, StepResult, EnvState
from environment import SessionManager
from tasks import TASK_DEFINITIONS, run_grader

from sre_platform import services as sre_services
from sre_platform.routes import apply_production_or_guardrails_step, build_platform_router
from sre_platform.state import STORE

# Gradio must stay available even if the UI module fails (e.g. missing optional paths in Docker).
gr = None
create_gradio_app = None
try:
    import gradio as gr
except ImportError:
    gr = None
if gr is not None:
    try:
        from ui import create_gradio_app
    except ImportError:
        create_gradio_app = None
        import warnings

        warnings.warn(
            "Gradio is installed but the Control Center UI did not import; /ui will be unavailable. "
            "On Hugging Face Spaces, ensure `training/rollout.py` is copied into the image (see Dockerfile).",
            stacklevel=1,
        )


def _csv_env(name: str, default: str = "") -> List[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


RUNTIME_API_KEY = secrets.token_urlsafe(32)
CONFIGURED_API_KEYS = set(_csv_env("CODEORGANISM_API_KEYS"))
if not CONFIGURED_API_KEYS:
    CONFIGURED_API_KEYS.add(RUNTIME_API_KEY)

AUTH_DISABLED = os.environ.get("CODEORGANISM_AUTH_DISABLED", "false").lower() in {"1", "true", "yes"}
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("CODEORGANISM_RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("CODEORGANISM_RATE_LIMIT_MAX", "120"))
_rate_limit_buckets: Dict[str, Deque[float]] = defaultdict(deque)

app = FastAPI(
    title="Autonomous SRE Control Center",
    description=(
        "OpenEnv environment where an agent performs incident response in a "
        "self-corrupting service sandbox."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env("CODEORGANISM_CORS_ORIGINS", "http://localhost:7860,http://127.0.0.1:7860"),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager holds all environment instances
sessions = SessionManager()

# Mount the interactive UI when Gradio is installed (may replace `app` reference).
if gr is not None and create_gradio_app is not None:
    demo = create_gradio_app()
    app = gr.mount_gradio_app(app, demo, path="/ui")

_console_dir = os.path.join(os.path.dirname(__file__), "static", "console")
if os.path.isdir(_console_dir):
    app.mount("/console", StaticFiles(directory=_console_dir, html=True), name="console")

# ── Request / Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "phase_1"

class GraderRequest(BaseModel):
    task_id: str
    actions: List[dict]


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> None:
    if AUTH_DISABLED:
        return

    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1]

    if not provided or not any(secrets.compare_digest(provided, key) for key in CONFIGURED_API_KEYS):
        raise HTTPException(status_code=401, detail="Valid API key required.")

    _enforce_rate_limit(request, provided)


def _enforce_rate_limit(request: Request, api_key: str) -> None:
    now = time.monotonic()
    bucket_key = f"{api_key}:{request.client.host if request.client else 'unknown'}"
    bucket = _rate_limit_buckets[bucket_key]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    bucket.append(now)


app.include_router(build_platform_router(sessions.get, require_api_key))

# ── Helper: resolve session from header ────────────────────────────────────────

def _get_env(session_id: Optional[str] = None):
    return sessions.get(session_id)

# ── OpenEnv Standard Endpoints ─────────────────────────────────────────────────

def _run_environment_step(action: Action, x_session_id: Optional[str] = None) -> StepResult:
    env = _get_env(x_session_id)
    blocked_or_pending, _ = apply_production_or_guardrails_step(env, action, x_session_id)
    if blocked_or_pending is not None:
        return blocked_or_pending
    result = env.step(action)
    st = STORE.get(x_session_id)
    sre_services.post_step_enrich(env, action, result, st)
    return result


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "autonomous-sre",
        "version": "1.0.0",
        "console_ui": "/console/",
        "legacy_gradio_ui": "/ui/",
        "platform_api": "/platform/session/state",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "autonomous-sre",
        "version": "1.0.0",
    }

@app.get("/metadata")
def metadata():
    return {
        "name": "autonomous-sre",
        "description": (
            "Autonomous SRE OpenEnv environment for incident-response policy training."
        ),
        "version": "1.0.0",
        "author": "Team Autonomous SRE",
        "tags": ["autonomous-sre", "openenv", "incident-response", "llm-agent"],
        "tasks": list(TASK_DEFINITIONS.keys()),
        "num_tasks": len(TASK_DEFINITIONS),
    }

@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }

# ── Core Environment Endpoints ─────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(
    req: Optional[ResetRequest] = None,
    task_id: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
    _auth: None = Depends(require_api_key),
):
    actual_task_id = task_id or (req.task_id if req else None) or "phase_1"
    if actual_task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {actual_task_id}")
    env = _get_env(x_session_id)
    obs = env.reset(actual_task_id)
    sre_services.on_env_reset(STORE.get(x_session_id), env)
    return obs

@app.post("/step", response_model=StepResult)
def step(action: Action, x_session_id: Optional[str] = Header(None), _auth: None = Depends(require_api_key)):
    return _run_environment_step(action, x_session_id)

@app.get("/state", response_model=EnvState)
def state(x_session_id: Optional[str] = Header(None), _auth: None = Depends(require_api_key)):
    env = _get_env(x_session_id)
    return env.state()

# ── Task & Grader Endpoints ────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    tasks = []
    for td in TASK_DEFINITIONS.values():
        tasks.append({
            "task_id": td.task_id,
            "name": td.name,
            "description": td.description,
            "difficulty": td.difficulty,
            "max_steps": td.max_steps,
            "scoring_summary": td.scoring_summary,
            "action_schema": Action.model_json_schema(),
        })
    return {"tasks": tasks}

@app.post("/grader")
def grader(req: GraderRequest, _auth: None = Depends(require_api_key)):
    if req.task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {req.task_id}")
    return run_grader(req.task_id, req.actions)

# ── MCP Integration (spec §48) ──────────────────────────────────────────────────

@app.get("/tools/list")
def list_mcp_tools():
    """Expose SRE protocols as standardized MCP Tools."""
    return {
        "tools": [
            {"name": "patch_file", "description": "Deploy a surgical hotfix (OLD|NEW format).", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "diff": {"type": "string"}}}},
            {"name": "run_tests", "description": "Execute cluster diagnostic suite.", "input_schema": {"type": "object", "properties": {}}},
            {"name": "rollback", "description": "Revert node to stable checkpoint.", "input_schema": {"type": "object", "properties": {"checkpoint_id": {"type": "string"}}}},
            {"name": "quarantine", "description": "Circuit-break a corrupt node.", "input_schema": {"type": "object", "properties": {"module": {"type": "string"}}}},
            {"name": "spawn_subagent", "description": "Delegate incident to parallel team.", "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}}},
            {"name": "request_expert", "description": "Consult high-fidelity expert oracle.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
            {"name": "emit_signal", "description": "Broadcast intent/metadata.", "input_schema": {"type": "object", "properties": {"signal_type": {"type": "string"}, "signal_data": {"type": "object"}}}},
        ]
    }

@app.post("/tools/call")
def call_mcp_tool(tool_call: dict, x_session_id: Optional[str] = Header(None), _auth: None = Depends(require_api_key)):
    """Universal MCP Tool Executor."""
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    env = _get_env(x_session_id)
    
    # Map MCP call to standard OpenEnv Action
    try:
        action = Action.model_validate({"action_type": name, **(args or {})})
        return _run_environment_step(action, x_session_id)
    except Exception as e:
        raise HTTPException(400, f"MCP Protocol Error: {e}")

# ── Session Management ─────────────────────────────────────────────────────────

@app.post("/sessions/create")
def create_session(_auth: None = Depends(require_api_key)):
    return {"session_id": sessions.create_session()}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, _auth: None = Depends(require_api_key)):
    if sessions.delete(session_id):
        STORE.drop(session_id)
        return {"deleted": session_id}
    raise HTTPException(404, "Session not found")

@app.get("/sessions")
def list_sessions(_auth: None = Depends(require_api_key)):
    return {"sessions": sessions.list_sessions()}

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    # Spaces and container runtimes require binding on 0.0.0.0.
    host = os.environ.get("HOST", "0.0.0.0")
    # Hugging Face (and other TLS terminators) talk HTTP to the container but set
    # X-Forwarded-Proto=https. Without this, Gradio emits http:// asset/API URLs →
    # mixed-content blocks, 503s, and "Unsafe attempt to load URL http://... from https://...".
    _fwd = os.environ.get("UVICORN_FORWARDED_ALLOW_IPS", "*")
    uvicorn.run(
        app,
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips=_fwd,
    )
