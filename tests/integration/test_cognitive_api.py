"""HTTP surface for the cognitive layer endpoints."""

from __future__ import annotations


def _write(client, content, namespace="ns"):
    return client.post(
        "/v1/memory/write",
        json={"content": content, "agent_id": "a", "session_id": "s", "namespace": namespace},
    )


def test_feedback_endpoint(client):
    res = _write(client, "the scheduler runs every five minutes")
    memory_id = res.json()["ids"][0]
    resp = client.post(
        "/v1/memory/feedback",
        json={"memory_id": memory_id, "namespace": "ns", "useful": True, "weight": 2.0},
    )
    assert resp.status_code == 204


def test_consolidate_endpoint(client):
    for _ in range(2):
        _write(client, "the report pipeline aggregates daily metrics", namespace="cns")
    resp = client.post("/v1/maintenance/consolidate", params={"namespace": "cns"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"scanned", "clusters", "insights"}


def test_conflicts_endpoint_lists(client):
    resp = client.get("/v1/conflicts", params={"namespace": "ns"})
    assert resp.status_code == 200
    assert "conflicts" in resp.json()


def test_graph_neighbors_endpoint(client):
    resp = client.get("/v1/graph/neighbors", params={"entity": "anything", "namespace": "ns"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity"] == "anything"
    assert body["edges"] == []


def test_resolve_missing_conflict_404(client):
    resp = client.post(
        "/v1/conflicts/nope/resolve", json={"winner_id": None}, params={"namespace": "ns"}
    )
    assert resp.status_code == 404
