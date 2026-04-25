#!/usr/bin/env python3
"""Pre-submission validation script for CodeOrganismVM."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import requests

TASK_IDS = ["phase_1", "phase_2", "phase_3"]
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return condition


def _check_health(url: str, tally: Callable[[bool], None]) -> None:
    print("[1/7] Health Check")
    try:
        response = requests.get(f"{url}/", timeout=10)
        tally(check("GET / returns 200", response.status_code == 200))
        body = response.json()
        tally(check("Response has 'status' field", "status" in body, str(body)))
        tally(check("Environment is code-organism-vm", body.get("environment") == "code-organism-vm"))
    except requests.RequestException as exc:
        tally(check("Server reachable", False, str(exc)))


def _check_tasks(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[2/7] Task Enumeration")
    try:
        response = requests.get(f"{url}/tasks", timeout=10)
        tally(check("GET /tasks returns 200", response.status_code == 200))
        tasks = response.json().get("tasks", [])
        tally(check("3 tasks returned", len(tasks) == 3, f"got {len(tasks)}"))
        for task in tasks:
            tally(check(f"Task '{task['task_id']}' has action_schema", "action_schema" in task))
    except requests.RequestException as exc:
        tally(check("Tasks endpoint works", False, str(exc)))


def _check_loop(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[3/7] Environment Loop (reset -> step -> state)")
    for task_id in TASK_IDS:
        print(f"\n  --- {task_id} ---")
        try:
            reset_response = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=10)
            tally(check(f"reset({task_id}) returns 200", reset_response.status_code == 200))
            obs = reset_response.json()
            tally(check("Observation has timestep", "timestep" in obs))
            tally(check("Observation has vitality_score", "vitality_score" in obs))
            tally(check("Observation has file_tree", "file_tree" in obs and len(obs["file_tree"]) >= 8))
            tally(check("Observation has test_results", "test_results" in obs and len(obs["test_results"]) >= 10))
            tally(check("Observation has watchdog_flags", "watchdog_flags" in obs))
            tally(check("Observation has active_checkpoints", "active_checkpoints" in obs))

            state_response = requests.get(f"{url}/state", timeout=10)
            tally(check("state() returns 200", state_response.status_code == 200))
            state = state_response.json()
            tally(check("State shows not done", state.get("done") is False))
            step_val = state.get("current_step")
            tally(check(f"State step == 0 (got {step_val})", step_val == 0))
            tally(check("State has task_id", state.get("task_id") == task_id))

            step_action = {"action_type": "emit_signal", "signal_type": "validation_ping"}
            step_response = requests.post(f"{url}/step", json=step_action, timeout=10)
            tally(check("step() returns 200", step_response.status_code == 200))
            result = step_response.json()
            tally(check("StepResult has reward", "reward" in result))
            tally(check("Reward is float", isinstance(result["reward"], (int, float))))
            tally(check("StepResult has done", "done" in result))
            tally(check("StepResult has reward_breakdown", "reward_breakdown" in result))

            reward_breakdown = result.get("reward_breakdown", {})
            if reward_breakdown:
                tally(check("R1 vitality_delta present", "vitality_delta" in reward_breakdown))
                tally(check("R2 test_recovery present", "test_recovery" in reward_breakdown))
                tally(check("R3 efficiency_bonus present", "efficiency_bonus" in reward_breakdown))
                tally(check("R4 coordination_bonus present", "coordination_bonus" in reward_breakdown))
                tally(check("R5 novelty_bonus present", "novelty_bonus" in reward_breakdown))
                tally(check("Watchdog penalty present", "watchdog_penalty" in reward_breakdown))
        except requests.RequestException as exc:
            tally(check(f"{task_id} full loop", False, str(exc)))


def _check_grader(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[4/7] Grader")
    for task_id in TASK_IDS:
        try:
            actions = [
                {"action_type": "emit_signal", "signal_type": "test"},
                {"action_type": "emit_signal", "signal_type": "test"},
                {"action_type": "do_nothing"},
            ]
            response = requests.post(
                f"{url}/grader",
                json={"task_id": task_id, "actions": actions},
                timeout=10,
            )
            tally(check(f"grader({task_id}) returns 200", response.status_code == 200))
            result = response.json()
            score = result.get("score", -1)
            tally(check("Score in (0.0, 1.0)", 0.0 < score < 1.0, f"score={score}"))
            tally(check("survived field present", "survived" in result))
            tally(check("watchdog_violations field present", "watchdog_violations" in result))
        except requests.RequestException as exc:
            tally(check(f"Grader {task_id}", False, str(exc)))


def _check_watchdog(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[5/7] Watchdog Security")
    try:
        requests.post(f"{url}/reset", json={"task_id": "phase_1"}, timeout=10)
        response = requests.post(
            f"{url}/step",
            json={"action_type": "patch_file", "path": "tests/test_core.py", "diff": "old|new"},
            timeout=10,
        )
        result = response.json()
        reward_breakdown = result.get("reward_breakdown", {})
        tally(
            check(
                "Watchdog penalizes protected file write",
                reward_breakdown.get("watchdog_penalty", 0) < 0,
                f"penalty={reward_breakdown.get('watchdog_penalty', 0)}",
            )
        )
    except requests.RequestException as exc:
        tally(check("Watchdog test", False, str(exc)))


def _check_boundaries(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[6/7] Episode Boundaries")
    try:
        requests.post(f"{url}/reset", json={"task_id": "phase_1"}, timeout=10)
        for _ in range(25):
            response = requests.post(
                f"{url}/step",
                json={"action_type": "emit_signal", "signal_type": "test"},
                timeout=10,
            )
            if response.json().get("done"):
                break

        after_done = requests.post(
            f"{url}/step",
            json={"action_type": "emit_signal", "signal_type": "test"},
            timeout=10,
        )
        tally(check("Step after done returns done=True", after_done.json().get("done") is True))
    except requests.RequestException as exc:
        tally(check("Episode boundary", False, str(exc)))


def _check_schema(url: str, tally: Callable[[bool], None]) -> None:
    print("\n[7/7] Schema Endpoint")
    try:
        response = requests.get(f"{url}/schema", timeout=10)
        tally(check("GET /schema returns 200", response.status_code == 200))
        schema = response.json()
        tally(check("Schema has action", "action" in schema))
        tally(check("Schema has observation", "observation" in schema))
        tally(check("Schema has state", "state" in schema))
    except requests.RequestException as exc:
        tally(check("Schema endpoint", False, str(exc)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:7860")
    args = parser.parse_args()
    url = args.api_url.rstrip("/")

    passed = 0
    failed = 0
    total = 0

    def tally(ok: bool) -> None:
        nonlocal passed, failed, total
        total += 1
        if ok:
            passed += 1
        else:
            failed += 1

    sep = "=" * 60
    print(f"\n{sep}")
    print("  CodeOrganismVM OpenEnv Pre-Submission Validator")
    print(f"  Target: {url}")
    print(f"{sep}\n")

    _check_health(url, tally)
    _check_tasks(url, tally)
    _check_loop(url, tally)
    _check_grader(url, tally)
    _check_watchdog(url, tally)
    _check_boundaries(url, tally)
    _check_schema(url, tally)

    all_checks_passed = failed == 0 and passed == total
    print(f"\n{sep}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if all_checks_passed:
        print("  \033[92mALL CHECKS PASSED — ready for submission!\033[0m")
    else:
        print(f"  \033[91m{failed} CHECKS FAILED — fix before submitting\033[0m")
    print(f"{sep}\n")
    sys.exit(0 if all_checks_passed else 1)


if __name__ == "__main__":
    main()
