"""The /metrics endpoint exposes Prometheus counters and histograms."""

from __future__ import annotations


def test_metrics_endpoint_reports_activity(client):
    client.post(
        "/memory/write",
        json={
            "content": "Cache invalidation happens on every deploy.",
            "agent_id": "a",
            "session_id": "s",
            "namespace": "ns",
        },
    )
    client.post("/memory/query", json={"query": "cache", "namespace": "ns"})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    assert "cb_memory_writes_total" in body
    assert "cb_queries_total" in body
    assert "cb_query_tokens_used" in body
    assert "cb_request_latency_seconds" in body
