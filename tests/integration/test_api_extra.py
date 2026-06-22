"""Batch write, listing, namespace-scoped keys and the async SDK client."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from context_bridge.api.app import create_app
from context_bridge.config import Settings
from context_bridge.sdk.client import AsyncContextBridgeClient


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "qdrant_url": ":memory:",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'extra.db'}",
    }
    base.update(overrides)
    return Settings(**base)


def test_batch_write_and_list(client):
    resp = client.post(
        "/v1/memory/write_batch",
        json={
            "items": [
                {
                    "content": "first batched note",
                    "agent_id": "a",
                    "session_id": "s",
                    "namespace": "batch",
                },
                {
                    "content": "second batched note",
                    "agent_id": "a",
                    "session_id": "s",
                    "namespace": "batch",
                },
            ]
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2

    listing = client.get("/v1/memory", params={"namespace": "batch", "limit": 50})
    assert listing.status_code == 200
    body = listing.json()
    assert len(body["chunks"]) >= 2
    assert all(c["namespace"] == "batch" for c in body["chunks"])


def test_namespace_scoped_key_enforced(tmp_path):
    settings = _settings(
        tmp_path,
        api_keys="team-a-key,team-b-key",
        api_key_namespaces='{"team-a-key": ["team-a"], "team-b-key": ["team-b"]}',
    )
    app = create_app(settings)
    with TestClient(app) as client:
        ok = client.post(
            "/v1/memory/query",
            json={"query": "x", "namespace": "team-a"},
            headers={"X-API-Key": "team-a-key"},
        )
        assert ok.status_code == 200

        forbidden = client.post(
            "/v1/memory/query",
            json={"query": "x", "namespace": "team-b"},
            headers={"X-API-Key": "team-a-key"},
        )
        assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_async_client_roundtrip(settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        sdk = AsyncContextBridgeClient()
        sdk._client = httpx.AsyncClient(transport=transport, base_url="http://test/v1")
        try:
            await sdk.remember(
                "an asynchronously stored fact about Saturn",
                agent_id="a",
                session_id="s",
                namespace="async",
            )
            result = await sdk.recall("Saturn", namespace="async")
            assert "Saturn" in result["context"]
        finally:
            await sdk.aclose()
