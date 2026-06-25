"""Failure memory: capturing lessons and surfacing them before similar work."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "lessons",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'lessons.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


def test_lesson_is_recalled_for_similar_situation(tmp_path):
    mgr = _manager(tmp_path, lessons_min_score=0.0)
    mgr.record_lesson(
        namespace="ns",
        trigger="deploying the payment service to production",
        guidance="run the migration before switching traffic",
        severity="high",
    )
    hits = mgr.relevant_lessons(query="about to deploy payment service", namespace="ns")
    assert hits
    assert hits[0]["guidance"] == "run the migration before switching traffic"
    assert hits[0]["severity"] == "high"


def test_failed_outcome_captures_lesson(tmp_path):
    mgr = _manager(tmp_path)
    out = mgr.record_outcome(
        session_id="run",
        namespace="ns",
        success=False,
        lesson="always validate input before persisting",
    )
    assert out["lesson_id"]
    assert mgr.list_lessons(namespace="ns")


def test_query_surfaces_lessons_as_guardrail(tmp_path):
    mgr = _manager(tmp_path, lessons_min_score=0.0)
    mgr.write(
        content="the deploy pipeline pushes to production",
        agent_id="a",
        session_id="s",
        namespace="ns",
    )
    mgr.record_lesson(
        namespace="ns",
        trigger="deploy pipeline production push",
        guidance="freeze deploys during incidents",
        severity="medium",
    )
    result = mgr.query(query="deploy pipeline", namespace="ns")
    assert result.lessons
    assert "Lessons from past mistakes" in result.context
    assert "freeze deploys during incidents" in result.context


def test_with_lessons_false_skips_guardrail(tmp_path):
    mgr = _manager(tmp_path, lessons_min_score=0.0)
    mgr.record_lesson(
        namespace="ns", trigger="anything at all", guidance="be careful", severity="low"
    )
    result = mgr.query(query="anything at all", namespace="ns", with_lessons=False)
    assert result.lessons == []
    assert "Lessons from past mistakes" not in result.context


def test_confirm_lesson_boosts_helpfulness(tmp_path):
    mgr = _manager(tmp_path)
    lid = mgr.record_lesson(namespace="ns", trigger="t", guidance="g")
    assert mgr.confirm_lesson(lid) is True
    assert mgr.confirm_lesson("missing") is False
    assert mgr.list_lessons(namespace="ns")[0]["times_helpful"] == 1


def test_preflight_bundles_lessons_and_procedures(tmp_path):
    mgr = _manager(tmp_path, lessons_min_score=0.0)
    mgr.record_lesson(
        namespace="ns", trigger="rotating credentials safely", guidance="revoke after issuing"
    )
    mgr.create_procedure(
        namespace="ns", title="rotate credentials", steps=["issue", "revoke"], created_by="ops"
    )
    brief = mgr.preflight(task="rotate credentials", namespace="ns")
    assert brief["lessons"]
    assert brief["procedures"]


def test_forget_clears_lessons(tmp_path):
    mgr = _manager(tmp_path)
    mgr.record_lesson(namespace="gone", trigger="t", guidance="g")
    assert mgr.list_lessons(namespace="gone")
    mgr.forget(namespace="gone")
    assert mgr.list_lessons(namespace="gone") == []
