"""Process-wide environment session manager (shared by FastAPI and Gradio UI)."""

from __future__ import annotations

from environment import SessionManager

sessions = SessionManager()
