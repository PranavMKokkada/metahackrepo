"""Per-session platform state (in-memory; survives until TTL with SessionManager)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PatchSuggestion:
    suggestion_id: str
    rank: int
    score: float
    path: str
    diff: str
    rationale: str


@dataclass
class PendingProductionBatch:
    batch_id: str
    created_at: float
    suggestions: List[PatchSuggestion]
    original_justification: str


@dataclass
class CICDStage:
    name: str
    status: str  # pending | running | success | failed
    detail: str
    ts: float


@dataclass
class CICDPipelineRun:
    run_id: str
    episode_id: int
    stages: List[CICDStage]
    current_stage_index: int = 0


@dataclass
class MemoryEntry:
    entry_id: str
    ts: float
    fault_signature: str
    strategy: str
    outcome: str
    task_id: str


@dataclass
class EvolutionSnapshot:
    ts: float
    step: int
    action_type: str
    cumulative_reward: float
    vitality: float


@dataclass
class PredictiveAlert:
    alert_id: str
    severity: str
    pattern: str
    recommendation: str
    ts: float


@dataclass
class SessionPlatformState:
    production_mode: bool = False
    rollback_confidence_min: float = 0.55
    restricted_paths_extra: List[str] = field(default_factory=list)
    safe_zones: List[str] = field(default_factory=lambda: ["schema/", "deployment_controller"])
    catastrophic_block: bool = True
    pending: Optional[PendingProductionBatch] = None
    cicd_runs: List[CICDPipelineRun] = field(default_factory=list)
    active_pipeline: Optional[CICDPipelineRun] = None
    memory_log: List[MemoryEntry] = field(default_factory=list)
    evolution: List[EvolutionSnapshot] = field(default_factory=list)
    business: Dict[str, Any] = field(default_factory=dict)
    ingested_logs: List[str] = field(default_factory=list)
    last_predictions: List[PredictiveAlert] = field(default_factory=list)
    incidents_total: int = 0
    incidents_auto_resolved: int = 0
    mttr_baseline_seconds: float = 45 * 60
    mttr_with_ai_seconds: Optional[float] = None
    step_timestamps: List[float] = field(default_factory=list)

    def touch(self) -> None:
        self.step_timestamps.append(time.time())


class PlatformStore:
    """Maps logical session ids (including default) to platform state."""

    def __init__(self) -> None:
        self._by_session: Dict[str, SessionPlatformState] = {}

    def get(self, session_id: Optional[str]) -> SessionPlatformState:
        sid = session_id or "default"
        if sid not in self._by_session:
            self._by_session[sid] = SessionPlatformState(
                business={
                    "mttr_baseline_min": 45.0,
                    "mttr_with_ai_min": None,
                    "incidents_auto_resolved_pct": 0.0,
                    "downtime_saved_seconds": 0.0,
                }
            )
        return self._by_session[sid]

    def reset_session_state(self, session_id: Optional[str]) -> None:
        sid = session_id or "default"
        self._by_session[sid] = SessionPlatformState(
            business={
                "mttr_baseline_min": 45.0,
                "mttr_with_ai_min": None,
                "incidents_auto_resolved_pct": 0.0,
                "downtime_saved_seconds": 0.0,
            }
        )

    def drop(self, session_id: Optional[str]) -> None:
        """Remove platform state when an API session is deleted (not the default session)."""
        sid = session_id or "default"
        if sid != "default" and sid in self._by_session:
            del self._by_session[sid]


STORE = PlatformStore()
