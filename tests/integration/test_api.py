"""Exercise the HTTP surface through FastAPI's TestClient."""

from __future__ import annotations


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readiness_endpoint(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["components"]["database"] == "ok"
    assert body["components"]["vector_store"] == "ok"


def test_write_query_timeline_flow(client):
    write = client.post(
        "/v1/memory/write",
        json={
            "content": "The incident was caused by an expired TLS certificate on the gateway.",
            "agent_id": "agent-sre",
            "session_id": "api-sess",
            "namespace": "ops",
        },
    )
    assert write.status_code == 200
    assert write.json()["stored"] == 1

    query = client.post(
        "/v1/memory/query",
        json={
            "query": "what caused the incident?",
            "namespace": "ops",
            "session_id": "api-sess",
            "token_budget": 150,
        },
    )
    assert query.status_code == 200
    body = query.json()
    assert "TLS certificate" in body["context"]
    assert body["tokens_used"] <= 150
    assert body["sources"][0]["agent_id"] == "agent-sre"

    timeline = client.get("/v1/sessions/api-sess/timeline")
    assert timeline.status_code == 200
    assert {e["kind"] for e in timeline.json()["episodes"]} >= {"write", "query"}


def test_get_missing_record_returns_404(client):
    resp = client.get("/v1/memory/does-not-exist")
    assert resp.status_code == 404
