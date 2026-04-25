"""
Standard OpenEnv Client for Autonomous SRE Control Center.
Provides a universal interface for interacting with the SRE Sandbox.
"""

from __future__ import annotations

import os
import requests
from typing import Optional, Dict, Any

from models import Action, Observation, StepResult, EnvState


class SREEnvClient:
    """Standard HTTP Client for SRE Environment (spec §9.3)."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def reset(self, task_id: str = "phase_1") -> Observation:
        """Provision a fresh cluster and start incident response."""
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        resp.raise_for_status()
        data = resp.json()
        return Observation(**data)

    def step(self, action: Action) -> StepResult:
        """Submit a remediation protocol to the cluster."""
        resp = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump()
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(**data)

    def state(self) -> EnvState:
        """Retrieve the current SLA and infrastructure state."""
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        data = resp.json()
        return EnvState(**data)

    def health(self) -> Dict[str, Any]:
        """Check cluster connection health."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    # Quick sanity check
    client = SREEnvClient()
    try:
        print("Connecting to Autonomous SRE Control Center...")
        health = client.health()
        print(f"Connection established. Environment Version: {health.get('version')}")
        
        print("Provisioning Phase 1 Cluster...")
        obs = client.reset("phase_1")
        print(f"Cluster Online. Initial SLA Index: {obs.vitality_score}%")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure 'python app.py' is running.")
