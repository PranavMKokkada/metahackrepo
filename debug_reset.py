import requests
url = "http://localhost:7860"
r = requests.post(f"{url}/reset", json={"task_id": "phase_1"})
print(f"Reset obs: {r.json()['timestep']}")
r = requests.get(f"{url}/state")
print(f"State step: {r.json()['current_step']}")
