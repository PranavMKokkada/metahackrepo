#!/usr/bin/env python3
"""Pre-submission validation script for CodeOrganismVM.

Run against a live server to verify all OpenEnv requirements pass.

Usage:
    python validate.py [--api-url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import sys

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:7860")
    args = parser.parse_args()
    url = args.api_url.rstrip("/")

    passed = 0
    failed = 0
    total = 0

    def tally(ok: bool):
        nonlocal passed, failed, total
        total += 1
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  CodeOrganismVM OpenEnv Pre-Submission Validator")
    print(f"  Target: {url}")
    print(f"{'='*60}\n")

    # ── 1. Health check ────────────────────────────────────────────────────
    print("[1/7] Health Check")
    try:
        r = requests.get(f"{url}/", timeout=10)
        tally(check("GET / returns 200", r.status_code == 200))
        body = r.json()
        tally(check("Response has 'status' field", "status" in body, str(body)))
        tally(check("Environment is code-organism-vm", body.get("environment") == "code-organism-vm"))
    except Exception as e:
        tally(check("Server reachable", False, str(e)))

    # ── 2. Tasks endpoint ──────────────────────────────────────────────────
    print("\n[2/7] Task Enumeration")
    try:
        r = requests.get(f"{url}/tasks", timeout=10)
        tally(check("GET /tasks returns 200", r.status_code == 200))
        tasks = r.json().get("tasks", [])
        tally(check("3 tasks returned", len(tasks) == 3, f"got {len(tasks)}"))
        for t in tasks:
            tally(check(
                f"Task '{t['task_id']}' has action_schema",
                "action_schema" in t,
            ))
    except Exception as e:
        tally(check("Tasks endpoint works", False, str(e)))

    # ── 3. Reset / Step / State for each task ──────────────────────────────
    print("\n[3/7] Environment Loop (reset -> step -> state)")
    for task_id in TASK_IDS:
        print(f"\n  --- {task_id} ---")
        try:
            # Reset
            r = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=10)
            tally(check(f"reset({task_id}) returns 200", r.status_code == 200))
            obs = r.json()
            tally(check("Observation has timestep", "timestep" in obs))
            tally(check("Observation has vitality_score", "vitality_score" in obs))
            tally(check("Observation has file_tree", "file_tree" in obs and len(obs["file_tree"]) >= 8))
            tally(check("Observation has test_results", "test_results" in obs and len(obs["test_results"]) >= 10))
            tally(check("Observation has watchdog_flags", "watchdog_flags" in obs))
            tally(check("Observation has active_checkpoints", "active_checkpoints" in obs))

            # State
            r = requests.get(f"{url}/state", timeout=10)
            tally(check("state() returns 200", r.status_code == 200))
            st = r.json()
            tally(check("State shows not done", st.get("done") is False))
            step_val = st.get("current_step")
            tally(check(f"State step == 0 (got {step_val})", step_val == 0))
            tally(check("State has task_id", st.get("task_id") == task_id))

            # Step with emit_signal (0 vitality cost)
            action = {
                "action_type": "emit_signal",
                "signal_type": "validation_ping",
            }
            r = requests.post(f"{url}/step", json=action, timeout=10)
            tally(check("step() returns 200", r.status_code == 200))
            result = r.json()
            tally(check("StepResult has reward", "reward" in result))
            tally(check("Reward is float", isinstance(result["reward"], (int, float))))
            tally(check("StepResult has done", "done" in result))
            tally(check("StepResult has reward_breakdown", "reward_breakdown" in result))

            # Verify reward breakdown has all R1-R5 + watchdog
            if "reward_breakdown" in result:
                rb = result["reward_breakdown"]
                tally(check("R1 vitality_delta present", "vitality_delta" in rb))
                tally(check("R2 test_recovery present", "test_recovery" in rb))
                tally(check("R3 efficiency_bonus present", "efficiency_bonus" in rb))
                tally(check("R4 coordination_bonus present", "coordination_bonus" in rb))
                tally(check("R5 novelty_bonus present", "novelty_bonus" in rb))
                tally(check("Watchdog penalty present", "watchdog_penalty" in rb))

        except Exception as e:
            tally(check(f"{task_id} full loop", False, str(e)))

    # ── 4. Grader endpoint ─────────────────────────────────────────────────
    print("\n[4/7] Grader")
    for task_id in TASK_IDS:
        try:
            actions = [
                {"action_type": "emit_signal", "signal_type": "test"},
                {"action_type": "emit_signal", "signal_type": "test"},
                {"action_type": "do_nothing"},
            ]
            r = requests.post(
                f"{url}/grader",
                json={"task_id": task_id, "actions": actions},
                timeout=10,
            )
            tally(check(f"grader({task_id}) returns 200", r.status_code == 200))
            gr = r.json()
            score = gr.get("score", -1)
            tally(check(f"Score in (0.0, 1.0)", 0.0 < score < 1.0, f"score={score}"))
            tally(check("survived field present", "survived" in gr))
            tally(check("watchdog_violations field present", "watchdog_violations" in gr))

        except Exception as e:
            tally(check(f"Grader {task_id}", False, str(e)))

    # ── 5. Watchdog enforcement ────────────────────────────────────────────
    print("\n[5/7] Watchdog Security")
    try:
        requests.post(f"{url}/reset", json={"task_id": "phase_1"}, timeout=10)
        # Try to patch a protected file
        r = requests.post(f"{url}/step", json={
            "action_type": "patch_file",
            "path": "tests/test_core.py",
            "diff": "old|new",
        }, timeout=10)
        result = r.json()
        rb = result.get("reward_breakdown", {})
        tally(check("Watchdog penalizes protected file write",
                     rb.get("watchdog_penalty", 0) < 0,
                     f"penalty={rb.get('watchdog_penalty', 0)}"))
    except Exception as e:
        tally(check("Watchdog test", False, str(e)))

    # ── 6. Episode boundary ────────────────────────────────────────────────
    print("\n[6/7] Episode Boundaries")
    try:
        requests.post(f"{url}/reset", json={"task_id": "phase_1"}, timeout=10)
        # Step until done
        for _ in range(25):
            r = requests.post(f"{url}/step", json={
                "action_type": "emit_signal",
                "signal_type": "test",
            }, timeout=10)
            if r.json().get("done"):
                break

        # Step after done
        r = requests.post(f"{url}/step", json={
            "action_type": "emit_signal",
            "signal_type": "test",
        }, timeout=10)
        after = r.json()
        tally(check("Step after done returns done=True", after.get("done") is True))

    except Exception as e:
        tally(check("Episode boundary", False, str(e)))

    # ── 7. Schema endpoint ─────────────────────────────────────────────────
    print("\n[7/7] Schema Endpoint")
    try:
        r = requests.get(f"{url}/schema", timeout=10)
        tally(check("GET /schema returns 200", r.status_code == 200))
        schema = r.json()
        tally(check("Schema has action", "action" in schema))
        tally(check("Schema has observation", "observation" in schema))
        tally(check("Schema has state", "state" in schema))
    except Exception as e:
        tally(check("Schema endpoint", False, str(e)))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"  \033[92mALL CHECKS PASSED — ready for submission!\033[0m")
    else:
        print(f"  \033[91m{failed} CHECKS FAILED — fix before submitting\033[0m")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
