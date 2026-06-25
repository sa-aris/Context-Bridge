"""HTTP surface for the health panel, auto-resolve and belief timeline."""

from __future__ import annotations


def _write(client, content, namespace="ins"):
    return client.post(
        "/v1/memory/write",
        json={"content": content, "agent_id": "a", "session_id": "s", "namespace": namespace},
    )


def test_namespace_health_endpoint(client):
    _write(client, "the queue drains every minute")
    resp = client.get("/v1/namespaces/ins/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["namespace"] == "ins"
    assert body["memories"] >= 1
    assert set(body["trust"]) == {"active", "demoted", "retired"}


def test_belief_timeline_endpoint(client):
    _write(client, "the worker pool has eight slots", namespace="bel")
    resp = client.get("/v1/namespaces/bel/beliefs", params={"query": "worker pool"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "worker pool"
    assert body["events"]


def test_auto_resolve_endpoint(client):
    resp = client.post("/v1/conflicts/auto-resolve", params={"namespace": "ins"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"resolved", "skipped"}
