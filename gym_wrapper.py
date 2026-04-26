"""Gymnasium-compatible wrapper for CodeOrganismVM.

Provides a standard Gymnasium interface for RL libraries.
Flattens the observation into a vector and provides a discrete action space.
"""

from __future__ import annotations

from typing import Optional, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from models import Action, CodeOrganismActionType
from data import CORE_PATH
from environment import CodeOrganismEnv

# Observation dimension calculation:
# - Vitality (1)
# - Timestep/Max (1)
# - # Files (1)
# - # Failing Tests (1)
# - # Quarantined (1)
# - # Checkpoints (1)
# - Test Results (binary vector of max 32 tests)
# - File Tree (binary vector of max 16 files, modified status)
OBS_DIM = 1 + 1 + 1 + 1 + 1 + 1 + 32 + 16


class CodeOrganismGymEnv(gym.Env):
    """Gymnasium-compatible wrapper for CodeOrganismVM.
    
    Action Space: Discrete(12)
      0: DO_NOTHING
      1: RUN_TESTS
      2: REQUEST_EXPERT
      3: ROLLBACK (last checkpoint)
      4-9: PATCH_FILE (predefined heuristics for top failing modules)
      10: QUARANTINE (top failing module)
      11: SPAWN_SUBAGENT (generic fix task)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "phase_1", render_mode: Optional[str] = None):
        super().__init__()
        self.task_id = task_id
        self.render_mode = render_mode
        self._env = CodeOrganismEnv()
        
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(12)

    def _get_obs_vec(self, obs: Any) -> np.ndarray:
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        
        # Scalars
        vec[idx] = obs.vitality_score
        idx += 1
        vec[idx] = obs.timestep / max(obs.max_steps, 1)
        idx += 1
        vec[idx] = len(obs.file_tree)
        idx += 1

        failing = [t for t in obs.test_results if t.status != "PASS"]
        vec[idx] = len(failing)
        idx += 1

        quarantined = [f for f in obs.file_tree if getattr(f, "is_quarantined", False)]
        vec[idx] = len(quarantined)
        idx += 1

        vec[idx] = len(obs.active_checkpoints)
        idx += 1
        
        # Test results (first 32)
        for i, t in enumerate(obs.test_results[:32]):
            vec[idx + i] = 1.0 if t.status == "PASS" else 0.0
        idx += 32
        
        # File modified status (first 16)
        for i, f in enumerate(obs.file_tree[:16]):
            vec[idx + i] = 1.0 if f.modified_at > 0 else 0.0
        idx += 16
        
        return vec

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs = self._env.reset(self.task_id)
        return self._get_obs_vec(obs), {}

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        target_path = self._resolve_target_path()
        action = self._action_from_index(action_idx, target_path)
        
        result = self._env.step(action)
        obs_vec = self._get_obs_vec(result.observation or self._env._make_observation())
        term = result.info.get("termination", "")
        truncated = bool(result.done and term == "timeout_death")
        terminated = bool(result.done and not truncated)
        return (obs_vec, result.reward, terminated, truncated, result.info)

    def _resolve_target_path(self) -> Optional[str]:
        failing_tests = [t for t in self._env._simulator.run_all_tests() if t.status != "PASS"]
        if not failing_tests:
            return None
        test_name = failing_tests[0].name
        return self._env._simulator.tests.get(test_name, {}).get("file")

    def _action_from_index(self, action_idx: int, target_path: Optional[str]) -> Action:
        action_type = CodeOrganismActionType.DO_NOTHING
        path = None
        diff = ""
        task = ""

        if action_idx == 1:
            action_type = CodeOrganismActionType.RUN_TESTS
        elif action_idx == 2:
            action_type = CodeOrganismActionType.REQUEST_EXPERT
        elif action_idx == 3:
            action_type = CodeOrganismActionType.ROLLBACK
            checkpoint_id = self._env._simulator.checkpoints[-1]["id"] if self._env._simulator.checkpoints else None
            return Action(
                action_type=action_type,
                checkpoint_id=checkpoint_id,
                justification=f"RL agent selected action {action_idx}",
            )
        elif 4 <= action_idx <= 9:
            action_type = CodeOrganismActionType.PATCH_FILE
            path = target_path or CORE_PATH
            diff = "retunr|return" if action_idx % 2 == 0 else "deaf |def "
        elif action_idx == 10:
            action_type = CodeOrganismActionType.QUARANTINE
            return Action(
                action_type=action_type,
                module=target_path,
                justification=f"RL agent selected action {action_idx}",
            )
        elif action_idx == 11:
            action_type = CodeOrganismActionType.SPAWN_SUBAGENT
            task = f"Fix {target_path}" if target_path else "Clean codebase"

        return Action(
            action_type=action_type,
            path=path,
            diff=diff,
            task=task,
            query="How to fix the failing tests?" if action_type == CodeOrganismActionType.REQUEST_EXPERT else None,
            justification=f"RL agent selected action {action_idx}",
        )

    def render(self):
        st = self._env.state()
        print(f"[{st.task_id}] Step {st.current_step} | Vitality: {st.vitality:.1f}% | Reward: {st.cumulative_reward:.4f}")
