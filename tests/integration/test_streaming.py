"""Server-Sent Events streaming recall."""

from __future__ import annotations


def test_query_stream_emits_chunk_and_done(client):
    client.post(
        "/v1/memory/write",
        json={
            "content": "The deploy was blocked by a failing database migration.",
            "agent_id": "agent-rel",
            "session_id": "s",
            "namespace": "stream",
        },
    )

    resp = client.post(
        "/v1/memory/query/stream",
        json={"query": "why was the deploy blocked?", "namespace": "stream", "token_budget": 200},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    assert "event: chunk" in body
    assert "event: done" in body
    assert "migration" in body


def test_query_stream_empty_namespace_still_finishes(client):
    resp = client.post(
        "/v1/memory/query/stream",
        json={"query": "anything", "namespace": "does-not-exist"},
    )
    assert resp.status_code == 200
    assert "event: done" in resp.text
