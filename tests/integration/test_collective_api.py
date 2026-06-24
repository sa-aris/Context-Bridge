"""HTTP surface for the collective-learning endpoints."""

from __future__ import annotations


def test_agents_outcomes_flow(client):
    client.post(
        "/v1/memory/write",
        json={
            "content": "deployed the hotfix",
            "agent_id": "sre",
            "session_id": "run",
            "namespace": "ns",
        },
    )
    outcome = client.post(
        "/v1/outcomes",
        json={"session_id": "run", "namespace": "ns", "success": True, "weight": 1.0},
    )
    assert outcome.status_code == 200
    assert outcome.json()["agents_credited"] == 1

    agents = client.get("/v1/agents", params={"namespace": "ns"})
    assert agents.status_code == 200
    assert any(a["agent_id"] == "sre" for a in agents.json()["agents"])


def test_procedures_endpoints(client):
    created = client.post(
        "/v1/procedures",
        json={
            "namespace": "ns",
            "title": "onboard service",
            "steps": ["a", "b"],
            "created_by": "x",
        },
    )
    assert created.status_code == 200
    pid = created.json()["id"]
    assert pid

    out = client.post(
        f"/v1/procedures/{pid}/outcome", json={"success": True}, params={"namespace": "ns"}
    )
    assert out.status_code == 204

    listed = client.get("/v1/procedures", params={"namespace": "ns"})
    assert listed.status_code == 200
    assert listed.json()["procedures"][0]["title"] == "onboard service"


def test_procedure_outcome_404(client):
    resp = client.post(
        "/v1/procedures/nope/outcome", json={"success": False}, params={"namespace": "ns"}
    )
    assert resp.status_code == 404
