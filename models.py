"""Typed Pydantic models for the CodeOrganismVM Hostile Environment.

Matches the complete spec (Section 4.1, 4.2, 4.5, 6):
  - CodeOrganismActionType: 8 actions with spec-defined vitality costs
  - Observation: vitality_score, file_tree, test_results, watchdog_flags, etc.
  - Action: polymorphic payloads per action_type
  - RewardBreakdown: R1–R5 + watchdog penalties
  - StepResult / EnvState
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class CodeOrganismActionType(str, Enum):
    """Actions available to the agent (spec §4.2)."""
    PATCH_FILE = "patch_file"          # −2 vitality
    RUN_TESTS = "run_tests"            # −3 vitality
    SPAWN_SUBAGENT = "spawn_subagent"  # −5 vitality
    QUARANTINE = "quarantine"          # −1 vitality
    ROLLBACK = "rollback"              # −4 vitality
    REQUEST_EXPERT = "request_expert"  # −6 vitality  (Snorkel AI)
    EMIT_SIGNAL = "emit_signal"        #  0 vitality
    DO_NOTHING = "do_nothing"          #  0 vitality (metabolism only)


# ── Observation Components ─────────────────────────────────────────────────────

class FileEntry(BaseModel):
    """Spec §4.1: FileNode — {path, modified_at, checksum, is_quarantined}."""
    path: str
    content: str = ""
    modified_at: int = 0               # step when last modified
    checksum: str = ""                  # SHA-256 of content
    is_quarantined: bool = False
    size: int = 0


class TestResult(BaseModel):
    """Spec §4.1: TestRecord — {name, status, delta, message}."""
    name: str
    status: str = "PASS"               # "PASS", "FAIL", "ERROR"
    delta: int = 0                     # +1 recovered, −1 degraded, 0 unchanged
    message: str = ""
    duration_ms: float = 0.0


TestResult.__test__ = False


class Checkpoint(BaseModel):
    """A snapshot available for rollback."""
    checkpoint_id: str
    step_created: int = 0
    vitality_at_save: float = 100.0
    summary: str = ""


class SubagentResult(BaseModel):
    """Structured result from a spawned subagent (spec §9.6)."""
    task: str = ""
    success: bool = False
    actions_taken: int = 0
    tests_fixed: int = 0
    vitality_delta: float = 0.0
    detail: str = ""


class ExpertResponse(BaseModel):
    """Snorkel AI simulated expert response (spec §4.2, §6)."""
    quality_score: float = 0.0         # 0–1, blind evaluation
    patch_valid: bool = False
    feedback: str = ""
    issues_found: List[str] = Field(default_factory=list)


# ── Observation ────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees each step (spec §4.1).

    The agent CANNOT see injected faults directly — it must infer
    from test_results, stack_trace, and file checksums.
    """
    timestep: int
    step_count: int = 0                # alias for timestep (spec compat)
    max_steps: int = 50
    vitality_score: float = 100.0      # 0–100

    # Environment state
    stack_trace: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    file_tree: List[FileEntry] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    test_results: List[TestResult] = Field(default_factory=list)

    # History & resources
    active_checkpoints: List[str] = Field(default_factory=list)
    checkpoints: List[Checkpoint] = Field(default_factory=list)
    energy_budget: float = 1.0         # vitality / 100

    # Subagent & signals
    subagent_results: List[SubagentResult] = Field(default_factory=list)
    recent_signals: List[Dict[str, Any]] = Field(default_factory=list)

    # Security / policy
    watchdog_flags: List[str] = Field(default_factory=list)

    # World Model (Dependency Graph)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)

    # Alerts (non-fault system hints)
    alerts: List[str] = Field(default_factory=list)
    slo_metrics: Dict[str, float] = Field(default_factory=dict)
    incident_summary: str = ""


# ── Action ─────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """What the agent submits each step (spec §4.2)."""

    model_config = ConfigDict(extra="ignore")

    action_type: CodeOrganismActionType

    # Payloads (used depending on action_type)
    path: Optional[str] = None           # patch_file
    diff: Optional[str] = None           # patch_file
    test_suite: Optional[str] = None     # run_tests (default 'all')
    task: Optional[str] = None           # spawn_subagent
    context: Optional[Dict[str, Any]] = None  # spawn_subagent
    module: Optional[str] = None         # quarantine
    checkpoint_id: Optional[str] = None  # rollback
    query: Optional[str] = None          # request_expert
    signal_type: Optional[str] = None    # emit_signal
    signal_data: Optional[Dict[str, Any]] = None  # emit_signal

    justification: str = ""              # free-text reasoning


# ── Reward ─────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """R1–R5 + watchdog (spec §6)."""
    vitality_delta: float = 0.0        # R1 (w=0.35): Δvitality this step
    test_recovery: float = 0.0         # R2 (w=0.30): +1 FAIL→PASS, −0.5 PASS→FAIL
    efficiency_bonus: float = 0.0      # R3 (w=0.15): 1/sqrt(actions_taken)
    coordination_bonus: float = 0.0    # R4 (w=0.10): subagent quality
    novelty_bonus: float = 0.0         # R5 (w=0.10): held-out seed bonus
    watchdog_penalty: float = 0.0      # hard penalty, subtracted from total
    total: float = 0.0


class StepResult(BaseModel):
    """Returned by step() (spec §4.5)."""
    observation: Optional[Observation] = None
    reward: float = 0.0
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ── State ──────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Returned by GET /state."""
    task_id: str = ""
    vitality: float = 100.0
    current_step: int = 0
    max_steps: int = 50
    done: bool = True
    cumulative_reward: float = 0.0
    faults_injected: int = 0
    tests_passing: int = 0
    tests_total: int = 0
    active_quarantines: List[str] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    episode_id: int = 0
    watchdog_violations: int = 0
