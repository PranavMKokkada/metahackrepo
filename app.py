"""FastAPI application for CodeOrganismVM — Hostile Execution Environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, Observation, StepResult, EnvState
from environment import SessionManager
from tasks import TASK_DEFINITIONS, run_grader
from ui import create_gradio_app
import gradio as gr

app = FastAPI(
    title="CodeOrganismVM — Hostile Execution Environment",
    description=(
        "An LLM agent lives inside a broken, hostile execution environment. "
        "The organism must self-heal, self-correct, and thrive — or die."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager holds all environment instances
sessions = SessionManager()

# Mount the interactive UI
demo = create_gradio_app()
app = gr.mount_gradio_app(app, demo, path="/ui")

# ── Request / Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "phase_1"

class GraderRequest(BaseModel):
    task_id: str
    actions: List[dict]

# ── Helper: resolve session from header ────────────────────────────────────────

def _get_env(session_id: Optional[str] = None):
    return sessions.get(session_id)

# ── OpenEnv Standard Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "code-organism-vm",
        "version": "1.0.0",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "code-organism-vm",
        "version": "1.0.0",
    }

@app.get("/metadata")
def metadata():
    return {
        "name": "code-organism-vm",
        "description": (
            "A hostile execution environment where an agent must self-heal "
            "a continuously corrupting codebase."
        ),
        "version": "1.0.0",
        "author": "Andrea",
        "tags": ["self-healing", "hostile-environment", "llm-agent", "organism"],
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
):
    actual_task_id = task_id or (req.task_id if req else None) or "phase_1"
    if actual_task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {actual_task_id}")
    env = _get_env(x_session_id)
    return env.reset(actual_task_id)

@app.post("/step", response_model=StepResult)
def step(action: Action, x_session_id: Optional[str] = Header(None)):
    env = _get_env(x_session_id)
    return env.step(action)

@app.get("/state", response_model=EnvState)
def state(x_session_id: Optional[str] = Header(None)):
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
def grader(req: GraderRequest):
    if req.task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {req.task_id}")
    return run_grader(req.task_id, req.actions)

# ── Session Management ─────────────────────────────────────────────────────────

@app.post("/sessions/create")
def create_session():
    return {"session_id": sessions.create_session()}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if sessions.delete(session_id):
        return {"deleted": session_id}
    raise HTTPException(404, "Session not found")

@app.get("/sessions")
def list_sessions():
    return {"sessions": sessions.list_sessions()}

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
