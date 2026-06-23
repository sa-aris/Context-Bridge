"""Role-based access control: per-key namespace globs and read/write scopes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from context_bridge.api.app import create_app
from context_bridge.config import Settings

_POLICIES = json.dumps(
    {
        "reader": {"namespaces": ["team-a*"], "permissions": ["read"]},
        "writer": {"namespaces": ["team-a"], "permissions": ["read", "write"]},
        "admin": {"namespaces": ["*"], "permissions": ["read", "write"]},
    }
)


def _client(tmp_path) -> TestClient:
    settings = Settings(
        qdrant_url=":memory:",
        embed_provider="hashing",
        embed_dim=128,
        rerank_provider="identity",
        working_provider="memory",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'rbac.db'}",
        api_keys="reader,writer,admin",
        api_key_policies=_POLICIES,
    )
    return TestClient(create_app(settings))


def _h(key: str) -> dict:
    return {"X-API-Key": key}


def test_read_only_key_can_query_but_not_write(tmp_path):
    with _client(tmp_path) as c:
        # glob match team-a* -> team-alpha is readable
        assert (
            c.post(
                "/v1/memory/query",
                json={"query": "x", "namespace": "team-alpha"},
                headers=_h("reader"),
            ).status_code
            == 200
        )
        # but no write permission
        assert (
            c.post(
                "/v1/memory/write",
                json={
                    "content": "x",
                    "agent_id": "a",
                    "session_id": "s",
                    "namespace": "team-alpha",
                },
                headers=_h("reader"),
            ).status_code
            == 403
        )


def test_namespace_glob_is_enforced(tmp_path):
    with _client(tmp_path) as c:
        # reader is scoped to team-a*, so team-b is denied
        assert (
            c.post(
                "/v1/memory/query", json={"query": "x", "namespace": "team-b"}, headers=_h("reader")
            ).status_code
            == 403
        )


def test_writer_can_write_its_namespace(tmp_path):
    with _client(tmp_path) as c:
        assert (
            c.post(
                "/v1/memory/write",
                json={"content": "x", "agent_id": "a", "session_id": "s", "namespace": "team-a"},
                headers=_h("writer"),
            ).status_code
            == 200
        )


def test_sweep_requires_global_write(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/v1/maintenance/sweep", headers=_h("admin")).status_code == 200
        assert c.post("/v1/maintenance/sweep", headers=_h("writer")).status_code == 403
