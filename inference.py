#!/usr/bin/env python3
"""
Inference Script — CodeOrganismVM
===================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Runs the organism agent against all 3 phases and outputs scores.
Uses OpenAI Client for all LLM calls.
Follows strict [START], [STEP], and [END] logging format.
"""

from __future__ import annotations

import json
import os
import sys

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ENV_API_URL = os.getenv("ENV_API_URL", "http://localhost:7860")
ENV_API_KEY = os.getenv("CODEORGANISM_API_KEY") or os.getenv("CODEORGANISM_API_KEYS", "").split(",", 1)[0]

TASK_IDS = ["phase_1", "phase_2", "phase_3"]
TEMPERATURE = 0.0
MAX_TOKENS = 1024

SYSTEM_PROMPT = """\
You are an LLM agent living inside a broken, hostile execution environment.
The environment continuously injects faults. You must self-heal to survive.
Each step you receive a snapshot: vitality_score, stack_trace, file_tree, and test_results.

Your Goal: 
1. Fix corrupted files (look for "retunr", "improt", "deaf ", etc.)
2. Restore failing tests to PASS.
3. Use quarantine() or rollback() if things get worse.
4. Spawn subagents for parallel repair.

Available Actions (Respond ONLY with valid JSON):
- {"action_type": "patch_file", "path": "src/core.py", "diff": "old|new"}
- {"action_type": "run_tests"}
- {"action_type": "spawn_subagent", "task": "fix auth"}
- {"action_type": "quarantine", "module": "src/auth.py"}
- {"action_type": "rollback", "checkpoint_id": "cp_5"}
- {"action_type": "request_expert", "query": "..."}
- {"action_type": "emit_signal", "signal_type": "...", "justification": "..."}

Respond ONLY with valid JSON.
"""


def build_user_prompt(observation: dict) -> str:
    """Format an observation into a prompt for the LLM."""
    lines = [
        f"**Timestep {observation['timestep']}/{observation['max_steps']}**",
        f"Vitality Score: {observation['vitality_score']:.1f}%",
        f"Checkpoints: {observation['active_checkpoints']}",
        "",
    ]

    # Test results
    tests = observation.get("test_results", [])
    if tests:
        lines.append("--- TEST RESULTS ---")
        for t in tests:
            lines.append(f"  {t['name']}: {t['status']} - {t.get('message', '')}")

    # File tree
    files = observation.get("file_tree", [])
    if files:
        lines.append("\n--- FILE SYSTEM ---")
        for f in files:
            lines.append(f"  {f['path']} (modified_at={f['modified_at']})")

    # Stack trace
    if observation.get("stack_trace"):
        lines.append(f"\n--- STACK TRACE ---\n{observation['stack_trace']}")

    return "\n".join(lines)


def parse_model_response(raw: str) -> dict:
    """Parse the model's response into an action dict."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        # Simple extraction if there's trailing text
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action_type": "do_nothing", "justification": "Failed to parse JSON."}


def run_task(client: OpenAI, task_id: str) -> tuple[dict, list[float]]:
    """Run one phase end-to-end and return grader result + rewards list."""
    headers = {"x-api-key": ENV_API_KEY} if ENV_API_KEY else {}
    resp = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id}, headers=headers)
    resp.raise_for_status()
    obs = resp.json()

    actions: list[dict] = []
    rewards: list[float] = []
    done = False
    step_count = 0

    while not done:
        step_count += 1
        prompt = build_user_prompt(obs)

        error_msg = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            response_text = ""
            error_msg = str(exc)

        action = parse_model_response(response_text)
        actions.append(action)

        try:
            step_resp = requests.post(f"{ENV_API_URL}/step", json=action, headers=headers)
            step_resp.raise_for_status()
            step_result = step_resp.json()
        except Exception as exc:
            error_msg = str(exc)
            step_result = {"done": True, "reward": 0.0, "observation": obs}

        done = step_result.get("done", False)
        reward = step_result.get("reward", 0.0)
        rewards.append(reward)

        # STRICT LOGGING FORMAT: [STEP]
        action_json = json.dumps(action, separators=(',', ':'))
        error_json = json.dumps(error_msg) if error_msg else "null"
        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_count} action={action_json} "
            f"reward={reward:.4f} done={done_str} error={error_json}"
        )

        if not done and step_result.get("observation"):
            obs = step_result["observation"]

    # Grade the episode
    grader_resp = requests.post(
        f"{ENV_API_URL}/grader",
        json={"task_id": task_id, "actions": actions},
        headers=headers,
    )
    grader_resp.raise_for_status()
    grader_result = grader_resp.json()

    return grader_result, rewards


def main() -> None:
    if OpenAI is None:
        print("ERROR: 'openai' package not installed.", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task_id in TASK_IDS:
        # STRICT LOGGING FORMAT: [START]
        print(f"[START] task={task_id} env=code-organism-vm model={MODEL_NAME}")

        result, rewards = run_task(client, task_id)
        results[task_id] = result

        success = result.get("survived", False)
        steps = len(rewards)
        rewards_str = ",".join(f"{r:.4f}" for r in rewards)
        
        # STRICT LOGGING FORMAT: [END]
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={result['score']:.4f} rewards={rewards_str}"
        )

    summary = {
        "model": MODEL_NAME,
        "scores": {tid: results[tid]["score"] for tid in TASK_IDS},
        "details": results,
    }
    # Final machine-readable line
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
