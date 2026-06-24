"""HTTP surface for ontology alignment and the collaboration-quality score."""

from __future__ import annotations


def _write(client, content, namespace="ns"):
    return client.post(
        "/v1/memory/write",
        json={"content": content, "agent_id": "a", "session_id": "s", "namespace": namespace},
    )


def test_alias_endpoint_registers_and_lists(client):
    resp = client.post(
        "/v1/graph/aliases",
        json={"namespace": "ns", "alias": "db one", "canonical": "Database One"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["registered"] is True
    assert {"alias": "db one", "canonical": "Database One"} in body["aliases"]


def test_align_endpoint(client):
    resp = client.post("/v1/graph/align", json={"namespace": "ns"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"groups_merged", "aliases_created"}


def test_quality_endpoint(client):
    _write(client, "the cache layer uses redis", namespace="qns")
    client.post(
        "/v1/memory/query",
        json={"query": "cache redis", "namespace": "qns", "session_id": "s"},
    )
    resp = client.get("/v1/quality", params={"namespace": "qns"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["writes"] == 1
    assert body["queries"] == 1
    assert 0.0 <= body["score"] <= 100.0
