"""Curriculum Manager for CodeOrganismVM.

Tracks agent performance across phases and manages advancement gates:
Phase 1 -> Phase 2 (>= 30% survival)
Phase 2 -> Phase 3 (>= 55% survival)
"""

from __future__ import annotations

import json
import os


class CurriculumManager:
    def __init__(self, history_file: str = "training/history.json"):
        self.history_file = history_file
        self.stats = {
            "current_phase": "phase_1",
            "episodes": [],
            "survival_rates": {"phase_1": 0.0, "phase_2": 0.0, "phase_3": 0.0}
        }
        self.load()

    def load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.stats = json.load(f)

    def save(self):
        with open(self.history_file, "w") as f:
            json.dump(self.stats, f, indent=2)

    def record_episode(self, phase: str, survived: bool, score: float):
        self.stats["episodes"].append({
            "phase": phase,
            "survived": survived,
            "score": score
        })
        
        # Keep window of last 100 episodes for survival rate
        relevant = [e for e in self.stats["episodes"] if e["phase"] == phase]
        if relevant:
            last_window = relevant[-100:]
            rate = sum(1 for e in last_window if e["survived"]) / len(last_window)
            self.stats["survival_rates"][phase] = rate
            
        self.check_gates()
        self.save()

    def check_gates(self):
        current = self.stats["current_phase"]
        rate = self.stats["survival_rates"].get(current, 0.0)
        
        if current == "phase_1" and rate >= 0.30:
            print(">>> ADVANCING TO PHASE 2")
            self.stats["current_phase"] = "phase_2"
        elif current == "phase_2" and rate >= 0.55:
            print(">>> ADVANCING TO PHASE 3")
            self.stats["current_phase"] = "phase_3"

    def get_active_phase(self) -> str:
        return self.stats["current_phase"]
