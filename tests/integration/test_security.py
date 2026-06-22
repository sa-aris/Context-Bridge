"""API-key auth and rate limiting on the data plane."""

from __future__ import annotations

from fastapi.testclient import TestClient

from context_bridge.api.app import create_app
from context_bridge.config import Settings


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "qdrant_url": ":memory:",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'sec.db'}",
    }
    base.update(overrides)
    return Settings(**base)


def test_missing_api_key_is_rejected(tmp_path):
    app = create_app(_settings(tmp_path, api_keys="secret-1,secret-2"))
    with TestClient(app) as client:
        resp = client.post("/v1/memory/query", json={"query": "anything"})
        assert resp.status_code == 401


def test_valid_api_key_is_accepted(tmp_path):
    app = create_app(_settings(tmp_path, api_keys="secret-1,secret-2"))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/memory/query",
            json={"query": "anything"},
            headers={"X-API-Key": "secret-2"},
        )
        assert resp.status_code == 200


def test_health_is_open_without_key(tmp_path):
    app = create_app(_settings(tmp_path, api_keys="secret-1"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_rate_limit_blocks_excess_requests(tmp_path):
    app = create_app(_settings(tmp_path, rate_limit_per_minute=2))
    with TestClient(app) as client:
        payload = {"query": "anything"}
        assert client.post("/v1/memory/query", json=payload).status_code == 200
        assert client.post("/v1/memory/query", json=payload).status_code == 200
        assert client.post("/v1/memory/query", json=payload).status_code == 429
