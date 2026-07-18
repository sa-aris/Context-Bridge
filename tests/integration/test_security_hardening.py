"""Regression tests for tenant isolation and fail-closed production settings."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from context_bridge.api.app import create_app
from context_bridge.api.security import _api_key_fingerprint
from context_bridge.config import Settings

_POLICIES = json.dumps(
    {
        "key-a": {"namespaces": ["team-a"], "permissions": ["read", "write"]},
        "key-b": {"namespaces": ["team-b"], "permissions": ["read", "write"]},
    }
)


def _client(tmp_path) -> TestClient:
    settings = Settings(
        qdrant_url=":memory:",
        embed_provider="hashing",
        embed_dim=128,
        rerank_provider="identity",
        working_provider="memory",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'security.db'}",
        api_keys="key-a,key-b",
        api_key_policies=_POLICIES,
    )
    return TestClient(create_app(settings))


def _h(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def _write(c: TestClient, *, key: str, namespace: str, session_id: str = "session") -> str:
    response = c.post(
        "/v1/memory/write",
        headers=_h(key),
        json={
            "content": f"private content for {namespace}",
            "agent_id": "agent",
            "session_id": session_id,
            "namespace": namespace,
        },
    )
    assert response.status_code == 200
    return response.json()["ids"][0]


def test_record_id_cannot_cross_namespace(tmp_path):
    with _client(tmp_path) as c:
        record_id = _write(c, key="key-a", namespace="team-a")

        assert c.get(f"/v1/memory/{record_id}", headers=_h("key-a")).status_code == 200
        assert c.get(f"/v1/memory/{record_id}", headers=_h("key-b")).status_code == 403
        assert c.delete(f"/v1/memory/{record_id}", headers=_h("key-b")).status_code == 403
        assert c.get(f"/v1/memory/{record_id}", headers=_h("key-a")).status_code == 200


def test_feedback_cannot_target_another_namespace(tmp_path):
    with _client(tmp_path) as c:
        record_id = _write(c, key="key-a", namespace="team-a")
        response = c.post(
            "/v1/memory/feedback",
            headers=_h("key-b"),
            json={
                "memory_id": record_id,
                "namespace": "team-b",
                "useful": True,
            },
        )
        assert response.status_code == 404


def test_timeline_and_turns_are_namespace_isolated(tmp_path):
    with _client(tmp_path) as c:
        _write(c, key="key-a", namespace="team-a", session_id="shared")
        _write(c, key="key-b", namespace="team-b", session_id="shared")

        for key, namespace, content in (
            ("key-a", "team-a", "turn-a"),
            ("key-b", "team-b", "turn-b"),
        ):
            response = c.post(
                "/v1/sessions/shared/turns",
                params={"namespace": namespace},
                headers=_h(key),
                json={"agent_id": "agent", "content": content},
            )
            assert response.status_code == 204

        timeline_a = c.get(
            "/v1/sessions/shared/timeline",
            params={"namespace": "team-a"},
            headers=_h("key-a"),
        )
        assert timeline_a.status_code == 200
        assert {episode["namespace"] for episode in timeline_a.json()["episodes"]} == {"team-a"}

        turns_a = c.get(
            "/v1/sessions/shared/turns",
            params={"namespace": "team-a"},
            headers=_h("key-a"),
        )
        turns_b = c.get(
            "/v1/sessions/shared/turns",
            params={"namespace": "team-b"},
            headers=_h("key-b"),
        )
        assert [turn["content"] for turn in turns_a.json()["turns"]] == ["turn-a"]
        assert [turn["content"] for turn in turns_b.json()["turns"]] == ["turn-b"]


def test_mixed_namespace_session_cannot_be_summarized_or_credited(tmp_path):
    with _client(tmp_path) as c:
        _write(c, key="key-a", namespace="team-a", session_id="shared")
        _write(c, key="key-b", namespace="team-b", session_id="shared")

        summary = c.post(
            "/v1/memory/summarize",
            headers=_h("key-a"),
            json={"session_id": "shared", "namespace": "team-a"},
        )
        outcome = c.post(
            "/v1/outcomes",
            headers=_h("key-a"),
            json={"session_id": "shared", "namespace": "team-a", "success": True},
        )
        assert summary.status_code == 409
        assert outcome.status_code == 409


def test_lesson_and_procedure_mutations_cannot_cross_namespace(tmp_path):
    with _client(tmp_path) as c:
        lesson = c.post(
            "/v1/lessons",
            headers=_h("key-a"),
            json={
                "namespace": "team-a",
                "trigger": "a failure",
                "guidance": "avoid it",
            },
        )
        procedure = c.post(
            "/v1/procedures",
            headers=_h("key-a"),
            json={"namespace": "team-a", "title": "safe plan", "steps": ["one"]},
        )
        assert lesson.status_code == 200
        assert procedure.status_code == 200

        lesson_id = lesson.json()["id"]
        procedure_id = procedure.json()["id"]
        assert (
            c.post(
                f"/v1/lessons/{lesson_id}/confirm",
                params={"namespace": "team-b"},
                headers=_h("key-b"),
            ).status_code
            == 404
        )
        assert (
            c.post(
                f"/v1/procedures/{procedure_id}/outcome",
                params={"namespace": "team-b"},
                headers=_h("key-b"),
                json={"success": True},
            ).status_code
            == 404
        )


def test_production_settings_refuse_fail_open_defaults():
    with pytest.raises(ValidationError):
        Settings(app_env="production")
    with pytest.raises(ValidationError):
        Settings(
            app_env="production",
            api_keys="x" * 32,
            rate_limit_per_minute=60,
            cors_allow_origins="*",
        )

    settings = Settings(
        app_env="production",
        api_keys="x" * 32,
        rate_limit_per_minute=60,
        cors_allow_origins="https://app.example.com",
    )
    assert settings.app_env == "production"


def test_api_key_fingerprint_never_contains_raw_key():
    raw = "high-entropy-secret-value"
    fingerprint = _api_key_fingerprint(raw)
    assert raw not in fingerprint
    assert len(fingerprint) == 64
