"""Policy rollout helpers for deterministic environment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
from typing import Any, Dict, List, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CodeOrganismEnv
from models import Action, CodeOrganismActionType, Observation, StepResult


@dataclass
class EpisodeTrace:
    policy: str
    task_id: str
    seed: int
    episode_id: int
    steps: int
    total_reward: float
    survived: bool
    thrived: bool
    termination: str
    final_vitality: float
    watchdog_violations: int
    tests_passing: int
    tests_total: int
    tests_recovered: int
    actions: List[Dict[str, Any]]
    reward_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "steps": self.steps,
            "total_reward": round(self.total_reward, 4),
            "survived": self.survived,
            "thrived": self.thrived,
            "termination": self.termination,
            "final_vitality": round(self.final_vitality, 4),
            "watchdog_violations": self.watchdog_violations,
            "tests_passing": self.tests_passing,
            "tests_total": self.tests_total,
            "tests_recovered": self.tests_recovered,
            "actions": self.actions,
            "reward_history": [round(v, 4) for v in self.reward_history],
        }


def build_policy_action(policy: str, env: CodeOrganismEnv, obs: Observation, step_idx: int) -> Action:
    if policy.startswith("stabilized:"):
        interval = int(policy.split(":", 1)[1])
        return _stabilized_action(obs, interval=interval)
    if policy == "noop":
        return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="No-op baseline.")
    if policy == "random":
        return _random_action(env, obs, step_idx)
    if policy == "heuristic":
        return _heuristic_action(env, obs)
    if policy == "stabilized":
        return _stabilized_action(obs, interval=4)
    if policy == "sft":
        return _sft_action(obs)
    raise ValueError(f"Unknown policy: {policy}")


def run_episode(policy: str, task_id: str, seed: int, max_steps: Optional[int] = None) -> EpisodeTrace:
    env = CodeOrganismEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    actions: List[Dict[str, Any]] = []
    rewards: List[float] = []
    tests_recovered = 0
    termination = "unknown"
    done = False

    limit = max_steps or 10_000
    step_idx = 0
    while not done and step_idx < limit:
        action = build_policy_action(policy, env, obs, step_idx)
        result = env.step(action)
        _record_step(actions, rewards, action, result, step_idx)
        tests_recovered += _count_recovered_tests(result)
        done = result.done
        termination = result.info.get("termination", termination)
        if result.observation is not None:
            obs = result.observation
        step_idx += 1

    state = env.state()
    survived = state.vitality > 0
    thrived = termination == "organism_thrival"
    return EpisodeTrace(
        policy=policy,
        task_id=task_id,
        seed=seed,
        episode_id=state.episode_id,
        steps=state.current_step,
        total_reward=state.cumulative_reward,
        survived=survived,
        thrived=thrived,
        termination=termination,
        final_vitality=state.vitality,
        watchdog_violations=state.watchdog_violations,
        tests_passing=state.tests_passing,
        tests_total=state.tests_total,
        tests_recovered=tests_recovered,
        actions=actions,
        reward_history=list(state.reward_history),
    )


def _record_step(
    actions: List[Dict[str, Any]],
    rewards: List[float],
    action: Action,
    result: StepResult,
    step_idx: int,
) -> None:
    rewards.append(result.reward)
    actions.append(
        {
            "step": step_idx + 1,
            "action_type": action.action_type.value,
            "path": action.path,
            "module": action.module,
            "signal_type": action.signal_type,
            "reward": round(result.reward, 4),
            "done": result.done,
        }
    )


def _count_recovered_tests(result: StepResult) -> int:
    if result.reward_breakdown is None:
        return 0
    recovery_score = result.reward_breakdown.test_recovery
    if recovery_score <= 0:
        return 0
    return int(round(recovery_score))


def _random_action(env: CodeOrganismEnv, obs: Observation, step_idx: int) -> Action:
    simulator = env._simulator
    if simulator is None:
        return Action(action_type=CodeOrganismActionType.DO_NOTHING)

    # Favor safe actions and avoid high failure-rate invalid patches.
    action_type_idx = (obs.timestep + step_idx) % 4
    if action_type_idx == 0:
        return Action(action_type=CodeOrganismActionType.RUN_TESTS, justification="Random baseline test run.")
    if action_type_idx == 1:
        return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="Random baseline idle.")
    if action_type_idx == 2:
        file_candidates = [p for p in simulator.files if p.endswith(".py")]
        if file_candidates:
            path = file_candidates[(obs.timestep + step_idx) % len(file_candidates)]
            content = simulator.files[path]
            needle = "return "
            if needle in content:
                return Action(
                    action_type=CodeOrganismActionType.PATCH_FILE,
                    path=path,
                    diff=f"{needle}|{needle}",
                    justification="Random baseline attempted no-op patch.",
                )
        return Action(action_type=CodeOrganismActionType.RUN_TESTS, justification="Random fallback.")
    if simulator.checkpoints:
        cp = simulator.checkpoints[-1]["id"]
        return Action(action_type=CodeOrganismActionType.ROLLBACK, checkpoint_id=cp, justification="Random rollback.")
    return Action(action_type=CodeOrganismActionType.DO_NOTHING)


def _heuristic_action(env: CodeOrganismEnv, obs: Observation) -> Action:
    simulator = env._simulator
    if simulator is None:
        return Action(action_type=CodeOrganismActionType.DO_NOTHING)

    # Prefer targeted repair of known injected faults.
    for fault in list(simulator.faults):
        if fault.target in simulator.files and isinstance(fault.original_value, str):
            current = simulator.files[fault.target]
            if current != fault.original_value:
                old = current
                new = fault.original_value
                return Action(
                    action_type=CodeOrganismActionType.PATCH_FILE,
                    path=fault.target,
                    diff=f"{old}|{new}",
                    justification=f"Heuristic restore for {fault.fault_type} on {fault.target}.",
                )

    failing_tests = [t for t in obs.test_results if t.status != "PASS"]
    if failing_tests:
        return Action(action_type=CodeOrganismActionType.RUN_TESTS, justification="Heuristic diagnostic run.")

    return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="Stable state monitoring.")


def _stabilized_action(obs: Observation, interval: int) -> Action:
    # Low-cost policy: preserve vitality and only run diagnostics periodically.
    if obs.timestep % max(1, interval) == 0:
        return Action(action_type=CodeOrganismActionType.RUN_TESTS, justification="Periodic health check.")
    return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="Stability-first control.")


def _load_sft_policy() -> Dict[str, Any]:
    policy_path = os.path.join(os.path.dirname(__file__), "policies", "sft_policy.json")
    if not os.path.exists(policy_path):
        return {"test_interval": 4}
    with open(policy_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    interval = int(data.get("test_interval", 4))
    return {"test_interval": max(1, interval)}


def _sft_action(obs: Observation) -> Action:
    cfg = _load_sft_policy()
    interval = cfg["test_interval"]
    if obs.timestep % interval == 0:
        return Action(
            action_type=CodeOrganismActionType.RUN_TESTS,
            justification=f"SFT policy diagnostic cadence every {interval} steps.",
        )
    if obs.vitality_score < 35:
        return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="SFT conserve vitality.")
    return Action(action_type=CodeOrganismActionType.DO_NOTHING, justification="SFT steady-state monitoring.")
