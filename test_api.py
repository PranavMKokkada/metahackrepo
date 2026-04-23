"""Quick API smoke test for CodeOrganismVM."""
import requests
import json

url = "http://localhost:7860"

# 1. Health
print("=== Health ===")
try:
    r = requests.get(f"{url}/ health") # Space deliberate to test normalization? No, fix.
    r = requests.get(f"{url}/health")
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print(f"Server not running: {e}")

# 2. Metadata
print("\n=== Metadata ===")
r = requests.get(f"{url}/metadata")
print(json.dumps(r.json(), indent=2))

# 3. Tasks
print("\n=== Tasks ===")
r = requests.get(f"{url}/tasks")
tasks = r.json()["tasks"]
for t in tasks:
    print(f"  {t['task_id']}: {t['name']} ({t['difficulty']})")

# 4. Full episode on phase_1
print("\n=== Running phase_1 lifecycle ===")
r = requests.post(f"{url}/reset", json={"task_id": "phase_1"})
obs = r.json()
print(f"  Reset OK, vitality={obs['vitality_score']}%")
print(f"  Files: {len(obs['file_tree'])}")
print(f"  Tests: {len(obs['test_results'])}")

done = False
step = 0
while not done and step < 5:
    # Just emit signal to avoid death too fast
    action = {
        "action_type": "emit_signal",
        "signal_type": "heartbeat",
        "justification": "Simulation heartbeat."
    }
    r = requests.post(f"{url}/step", json=action)
    result = r.json()
    done = result["done"]
    print(f"  Step {step}: reward={result['reward']:.4f} vitality={result['observation']['vitality_score']}% done={done}")
    step += 1

# 5. Grader
print("\n=== Grader ===")
actions = [
    {"action_type": "emit_signal", "signal_type": "test", "justification": "Replay 1"},
    {"action_type": "do_nothing", "justification": "Replay 2"},
]
r = requests.post(f"{url}/grader", json={"task_id": "phase_1", "actions": actions})
gr = r.json()
print(f"  Score: {gr['score']}")
print(f"  Survived: {gr['survived']}")

print("\n=== ALL API CHECKS PASSED ===")
