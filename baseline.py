#!/usr/bin/env python3
"""Baseline inference script for CodeOrganismVM."""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

TASK_IDS = ["phase_1", "phase_2", "phase_3"]

SYSTEM_PROMPT = """\
You are an LLM agent living inside a broken, hostile execution environment.
The environment continuously injects faults. You must self-heal to survive.
Each step you receive a snapshot: vitality_score, stack_trace, file_tree, env_vars, and test_results.

Your Goal: 
1. Fix corrupted files (look for "retunr", syntax errors, etc.)
2. Restore failing tests to PASS.
3. Use quarantine() or rollback() if things get worse.
4. Spawn subagents for parallel repair.

Available Actions:
- patch_file(path, diff): e.g. {"action_type": "patch_file", "path": "src/core.py", "diff": "retunr|return"}
- run_tests(test_suite): {"action_type": "run_tests"}
- spawn_subagent(task): {"action_type": "spawn_subagent", "task": "fix auth"}
- quarantine(module): {"action_type": "quarantine", "module": "src"}
- rollback(checkpoint_id): {"action_type": "rollback", "checkpoint_id": "cp_..."}
- emit_signal(signal_type): {"action_type": "emit_signal", "signal_type": "repair"}
- do_nothing(): {"action_type": "do_nothing"}

Respond ONLY with valid JSON.
"""

def build_user_prompt(observation: dict) -> str:
    lines = [
        f"Timestep {observation['timestep']}/{observation['max_steps']}",
        f"Vitality: {observation['vitality_score']}%",
        f"Energy: {observation['energy_budget']:.2f}",
        "",
    ]
    
    if observation.get("test_results"):
        lines.append("Test Results:")
        for t in observation["test_results"]:
            lines.append(f"  {t['name']}: {t['status']} - {t['message']}")
            
    if observation.get("file_tree"):
        lines.append("\nFile Tree:")
        for f in observation["file_tree"]:
            corrupt = " (CORRUPTED)" if f.get("is_corrupted") else ""
            lines.append(f"  - {f['path']}{corrupt}")
            
    return "\n".join(lines)


def run_task(client: OpenAI, api_url: str, task_id: str, model: str) -> dict:
    resp = requests.post(f"{api_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    actions: list[dict] = []
    done = False
    
    while not done:
        prompt = build_user_prompt(obs)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
        )

        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            action = {"action_type": "do_nothing"}

        actions.append(action)
        step_resp = requests.post(f"{api_url}/step", json=action)
        step_resp.raise_for_status()
        step_result = step_resp.json()

        done = step_result["done"]
        if not done and step_result.get("observation"):
            obs = step_result["observation"]

    grader_resp = requests.post(
        f"{api_url}/grader", json={"task_id": task_id, "actions": actions}
    )
    grader_resp.raise_for_status()
    return grader_resp.json()


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for CodeOrganismVM")
    parser.add_argument("--api-url", default="http://localhost:7860")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if OpenAI is None:
        print("openai package not installed", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results = {}
    for task_id in TASK_IDS:
        print(f"Running {task_id}...", file=sys.stderr)
        result = run_task(client, args.api_url, task_id, args.model)
        results[task_id] = result
        print(f"  {task_id}: score={result['score']} survived={result['survived']} vitality={result['final_vitality']}", file=sys.stderr)

    summary = {
        "model": args.model,
        "details": results,
    }
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
