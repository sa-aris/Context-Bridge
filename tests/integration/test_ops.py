"""Scheduled maintenance, event emission, and namespace import/export."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "ops",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


class _RecordingEmitter:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def emit(self, event_type: str, namespace: str, data: dict) -> None:
        self.events.append((event_type, namespace, data))


# -- events -----------------------------------------------------------------
def test_lesson_created_emits_event(tmp_path):
    mgr = _manager(tmp_path)
    mgr.events = _RecordingEmitter()
    mgr.record_lesson(namespace="ns", trigger="t", guidance="g", severity="high")
    types = [e[0] for e in mgr.events.events]
    assert "lesson.created" in types


def test_conflict_lifecycle_emits_events(tmp_path):
    mgr = _manager(
        tmp_path, detect_contradictions=True, contradiction_similarity=0.5, dedup_threshold=0.99
    )
    rec = _RecordingEmitter()
    mgr.events = rec
    mgr.write(content="the token expires in one hour", agent_id="a", session_id="s", namespace="ns")
    mgr.write(
        content="the token does not expire in one hour",
        agent_id="b",
        session_id="s",
        namespace="ns",
    )
    assert any(e[0] == "conflict.opened" for e in rec.events)

    conflict = mgr.list_conflicts(namespace="ns")[0]
    mgr.resolve_conflict(conflict["id"], winner_id=conflict["memory_id_a"])
    assert any(e[0] == "conflict.resolved" for e in rec.events)


# -- scheduled maintenance --------------------------------------------------
def test_run_maintenance_resolves_across_namespaces(tmp_path):
    mgr = _manager(
        tmp_path, detect_contradictions=True, contradiction_similarity=0.5, dedup_threshold=0.99
    )
    mgr.write(content="the api rate limit is 100 rps", agent_id="a", session_id="s", namespace="ns")
    mgr.write(
        content="the api rate limit is not 100 rps", agent_id="b", session_id="s", namespace="ns"
    )
    conflict = mgr.list_conflicts(namespace="ns")[0]
    mgr.record_feedback(memory_id=conflict["memory_id_a"], namespace="ns", useful=True, weight=8)
    mgr.record_feedback(memory_id=conflict["memory_id_b"], namespace="ns", useful=False, weight=8)

    result = mgr.run_maintenance(auto_resolve=True, min_gap=0.1)
    assert result["namespaces"] >= 1
    assert result["conflicts_resolved"] >= 1
    assert mgr.list_conflicts(namespace="ns", status="open") == []


# -- portability ------------------------------------------------------------
def test_export_then_import_round_trips_a_namespace(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="the cache uses an LRU policy", agent_id="a", session_id="s", namespace="src")
    mgr.write(content="the queue is backed by redis", agent_id="a", session_id="s", namespace="src")
    mgr.record_lesson(namespace="src", trigger="cache sizing", guidance="cap the cache at 1GB")
    mgr.create_procedure(namespace="src", title="warm the cache", steps=["load", "verify"])

    dump = mgr.export_namespace(namespace="src")
    assert len(dump["memories"]) == 2
    assert dump["lessons"] and dump["procedures"]

    counts = mgr.import_namespace(namespace="dst", payload=dump)
    assert counts == {"memories": 2, "lessons": 1, "procedures": 1}

    recalled = mgr.query(query="redis queue", namespace="dst", with_lessons=False)
    assert recalled.chunks
    assert mgr.list_lessons(namespace="dst")
    assert mgr.list_procedures(namespace="dst")
