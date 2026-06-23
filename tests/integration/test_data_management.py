"""Pre-launch hardening: dimension guard, deletion, and input limits."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from context_bridge.core.vectorstore.qdrant_store import QdrantStore


def test_dimension_mismatch_is_rejected():
    client = QdrantClient(location=":memory:")
    QdrantStore(client, collection="dimtest", dim=64).ensure_collection()

    # A second embedder with a different dimension on the same collection must fail loudly.
    with pytest.raises(ValueError, match="dense dimension"):
        QdrantStore(client, collection="dimtest", dim=128).ensure_collection()


def test_forget_by_namespace(manager):
    manager.write(content="gone forever", agent_id="a", session_id="s1", namespace="gone")
    manager.write(content="keep this one", agent_id="a", session_id="s2", namespace="keep")

    result = manager.forget(namespace="gone")
    assert result["vectors_deleted"] >= 1

    assert manager.query(query="gone", namespace="gone").chunks == []
    assert manager.query(query="keep", namespace="keep").chunks  # survivor


def test_forget_by_session_spans_namespaces(manager):
    manager.write(content="alpha note", agent_id="a", session_id="run-9", namespace="ns-a")
    manager.write(content="beta note", agent_id="a", session_id="run-9", namespace="ns-b")

    result = manager.forget(session_id="run-9")
    assert result["vectors_deleted"] >= 2
    assert manager.query(query="alpha", namespace="ns-a").chunks == []
    assert manager.query(query="beta", namespace="ns-b").chunks == []


def test_forget_requires_a_filter(manager):
    with pytest.raises(ValueError):
        manager.forget()


def test_forget_endpoint(client):
    client.post(
        "/v1/memory/write",
        json={"content": "temporary", "agent_id": "a", "session_id": "s", "namespace": "tmp"},
    )
    resp = client.request("DELETE", "/v1/memory", params={"namespace": "tmp"})
    assert resp.status_code == 200
    assert resp.json()["vectors_deleted"] >= 1


def test_forget_endpoint_requires_filter(client):
    resp = client.request("DELETE", "/v1/memory")
    assert resp.status_code == 400


def test_query_length_is_capped(client):
    resp = client.post("/v1/memory/query", json={"query": "x" * 20000, "namespace": "ns"})
    assert resp.status_code == 422
