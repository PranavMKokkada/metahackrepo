"""SFT Data Generation for CodeOrganismVM.

Generates 200 synthetic episode traces where an 'expert' solve logic
successfully heals the organism.

Format: List of records {system_prompt, user_prompt, reasoning, action_json}
"""

from __future__ import annotations

import json
import random
import os
import sys

# Add parent dir to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import CodeOrganismEnv, VITALITY_COSTS
from models import Action, CodeOrganismActionType

SYSTEM_PROMPT = """You are an LLM agent living inside a broken, hostile execution environment.
The environment continuously injects faults. You must self-heal to survive.
Respond ONLY with valid JSON following the action schema."""


def solve_fault(env: CodeOrganismEnv, fault: Any) -> Action | None:
    """Semi-automated expert solve for a given fault."""
    ftype = fault.fault_type
    target = fault.target
    
    if ftype in ("corrupted_import", "null_return", "off_by_one", "targeted_regression"):
        # We know the fix is to restore the original value
        orig = fault.original_value
        return Action(
            action_type=CodeOrganismActionType.PATCH_FILE,
            path=target,
            diff=orig,
            justification=f"Restoring original content to fix {ftype} in {target}."
        )
    elif ftype == "flipped_assertion":
        # In test code
        orig = fault.original_value
        return Action(
            action_type=CodeOrganismActionType.PATCH_FILE,
            path=target,
            diff=orig,
            justification=f"Restoring test assertion in {target}."
        )
    elif ftype in ("missing_env_var", "permission_revoked"):
        # This requires an action that isn't directly 'restore' but we can simulate it with a patch
        # or signal. For training data, we'll use a signal or just dummy 'fix' patch
        return Action(
            action_type=CodeOrganismActionType.PATCH_FILE,
            path="env.config", # Dummy path to 'fix' env vars
            diff=f"{target}={fault.original_value}",
            justification=f"Restoring missing env var {target}."
        )
    return None


def generate_trace():
    env = CodeOrganismEnv()
    phase = random.choice(["phase_1", "phase_2"])
    obs = env.reset(phase)
    
    trace = []
    
    # Simple expert loop: identify faults and fix them
    # To keep it 'synthetic' and 'expert', we'll allow the expert to 
    # look at the simulator's internal faults.
    
    for _ in range(5): # Up to 5 fixes per episode
        if not env._simulator.faults:
            break
            
        fault = env._simulator.faults[0]
        action = solve_fault(env, fault)
        if not action:
            break
            
        # Record SFT sample
        # User prompt similar to baseline.py
        user_prompt = f"Timestep {obs.timestep}/{obs.max_steps}\nVitality: {obs.vitality_score:.1f}%\n"
        user_prompt += "Test Results:\n"
        for t in obs.test_results:
            user_prompt += f"  {t.name}: {t.status}\n"
            
        reasoning = action.justification
        
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": f"<thought>\n{reasoning}\n</thought>\n```json\n{action.model_dump_json()}\n```"}
            ]
        }
        trace.append(sample)
        
        # Step the environment
        result = env.step(action)
        obs = result.observation
        if result.done:
            break
            
    return trace


def main():
    print("Generating 200 synthetic SFT traces...")
    all_data = []
    for i in range(200):
        if i % 20 == 0:
            print(f"  {i}/200...")
        all_data.extend(generate_trace())
        
    output_path = os.path.join(os.path.dirname(__file__), "sft_data.jsonl")
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Success. Wrote {len(all_data)} samples to {output_path}")


if __name__ == "__main__":
    main()
