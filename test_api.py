"""FastAPI contract tests for CodeOrganismVM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import RUNTIME_API_KEY, app


client = TestClient(app)
AUTH_HEADERS = {"x-api-key": RUNTIME_API_KEY}


def test_public_metadata_endpoints():
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "healthy"

    metadata = client.get("/metadata").json()
    assert metadata["name"] == "code-organism-vm"

    tasks = client.get("/tasks").json()["tasks"]
    assert {task["task_id"] for task in tasks} == {"phase_1", "phase_2", "phase_3"}


def test_mutating_endpoints_require_api_key():
    response = client.post("/reset", json={"task_id": "phase_1"})
    assert response.status_code == 401


def test_reset_step_state_loop():
    reset_response = client.post("/reset", json={"task_id": "phase_1"}, headers=AUTH_HEADERS)
    assert reset_response.status_code == 200
    obs = reset_response.json()
    assert obs["vitality_score"] == 100.0
    assert obs["file_tree"]
    assert obs["test_results"]

    step_response = client.post(
        "/step",
        json={"action_type": "emit_signal", "signal_type": "heartbeat"},
        headers=AUTH_HEADERS,
    )
    assert step_response.status_code == 200
    result = step_response.json()
    assert "reward" in result
    assert "reward_breakdown" in result

    state_response = client.get("/state", headers=AUTH_HEADERS)
    assert state_response.status_code == 200
    assert state_response.json()["current_step"] == 1


def test_mcp_tool_call_maps_to_action():
    client.post("/reset", json={"task_id": "phase_1"}, headers=AUTH_HEADERS)
    response = client.post(
        "/tools/call",
        json={"name": "emit_signal", "arguments": {"signal_type": "mcp_ping"}},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["info"]["action_result"]["result"] == "signal_emitted"


def test_grader_and_sessions():
    create_response = client.post("/sessions/create", headers=AUTH_HEADERS)
    assert create_response.status_code == 200
    session_id = create_response.json()["session_id"]

    list_response = client.get("/sessions", headers=AUTH_HEADERS)
    assert session_id in list_response.json()["sessions"]

    delete_response = client.delete(f"/sessions/{session_id}", headers=AUTH_HEADERS)
    assert delete_response.status_code == 200

    grader_response = client.post(
        "/grader",
        json={
            "task_id": "phase_1",
            "actions": [{"action_type": "emit_signal", "signal_type": "test"}],
        },
        headers=AUTH_HEADERS,
    )
    assert grader_response.status_code == 200
    assert "score" in grader_response.json()
