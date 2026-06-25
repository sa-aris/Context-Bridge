"""HTTP surface for lesson distillation and recall explainability."""

from __future__ import annotations


def test_distill_endpoint(client):
    resp = client.post("/v1/lessons/distill", params={"namespace": "dns"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"scanned", "clusters", "lessons_created"}


def test_query_chunks_include_reason(client):
    client.post(
        "/v1/memory/write",
        json={
            "content": "the api gateway caches responses",
            "agent_id": "a",
            "session_id": "s",
            "namespace": "ex",
        },
    )
    resp = client.post("/v1/memory/query", json={"query": "gateway caches", "namespace": "ex"})
    assert resp.status_code == 200
    chunks = resp.json()["chunks"]
    assert chunks
    assert "reason" in chunks[0]
    assert "signals" in chunks[0]
    assert chunks[0]["reason"]
