"""Health panel, auto conflict resolution, and belief timeline (memory diff)."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "insights",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'insights.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


def _conflicting(tmp_path):
    return _manager(
        tmp_path, detect_contradictions=True, contradiction_similarity=0.5, dedup_threshold=0.99
    )


# -- auto conflict resolution ----------------------------------------------
def test_auto_resolve_picks_the_authoritative_memory(tmp_path):
    mgr = _conflicting(tmp_path)
    mgr.write(content="the api rate limit is 100 rps", agent_id="a", session_id="s", namespace="ns")
    mgr.write(
        content="the api rate limit is not 100 rps", agent_id="b", session_id="s", namespace="ns"
    )
    conflict = mgr.list_conflicts(namespace="ns")[0]
    mgr.record_feedback(memory_id=conflict["memory_id_a"], namespace="ns", useful=True, weight=8)
    mgr.record_feedback(memory_id=conflict["memory_id_b"], namespace="ns", useful=False, weight=8)

    out = mgr.auto_resolve_conflicts(namespace="ns", min_gap=0.1)
    assert out["resolved"] == 1
    assert mgr.list_conflicts(namespace="ns", status="open") == []
    # The discredited side was decayed by belief revision.
    assert mgr.get(conflict["memory_id_b"]).provenance.confidence < 1.0


def test_auto_resolve_leaves_ambiguous_conflicts_for_humans(tmp_path):
    mgr = _conflicting(tmp_path)
    mgr.write(content="the build takes ten minutes", agent_id="a", session_id="s", namespace="ns")
    mgr.write(
        content="the build does not take ten minutes", agent_id="b", session_id="s", namespace="ns"
    )
    out = mgr.auto_resolve_conflicts(namespace="ns", min_gap=0.3)
    assert out["resolved"] == 0
    assert out["skipped"] >= 1
    assert mgr.list_conflicts(namespace="ns", status="open")


# -- belief timeline (memory diff) -----------------------------------------
def test_belief_timeline_marks_demoted_memory(tmp_path):
    mgr = _conflicting(tmp_path)
    mgr.write(content="the feature flag is on", agent_id="a", session_id="s", namespace="ns")
    mgr.write(content="the feature flag is not on", agent_id="b", session_id="s", namespace="ns")
    conflict = mgr.list_conflicts(namespace="ns")[0]
    mgr.resolve_conflict(conflict["id"], winner_id=conflict["memory_id_a"])

    events = mgr.belief_timeline(query="feature flag", namespace="ns")
    assert events
    by_id = {e["id"]: e for e in events}
    assert by_id[conflict["memory_id_b"]]["status"] in {"demoted", "retired"}
    assert all(e["date"] for e in events)


# -- health panel -----------------------------------------------------------
def test_namespace_health_summarizes_the_pool(tmp_path):
    mgr = _manager(tmp_path)
    for i in range(3):
        mgr.write(
            content=f"distinct memory number {i}", agent_id="a", session_id="s", namespace="ns"
        )
    mgr.query(query="distinct memory", namespace="ns", session_id="s")

    health = mgr.namespace_health(namespace="ns")
    assert health["memories"] >= 3
    assert health["trust"]["active"] >= 3
    assert health["writes"] >= 3
    assert health["queries"] >= 1
    assert 0.0 <= health["quality_score"] <= 100.0
    assert health["avg_confidence"] == 1.0


def test_health_reflects_demotion(tmp_path):
    mgr = _conflicting(tmp_path)
    mgr.write(
        content="the cron job is enabled in production",
        agent_id="a",
        session_id="s",
        namespace="ns",
    )
    mgr.write(
        content="the cron job is not enabled in production",
        agent_id="b",
        session_id="s",
        namespace="ns",
    )
    conflict = mgr.list_conflicts(namespace="ns")[0]
    mgr.resolve_conflict(conflict["id"], winner_id=conflict["memory_id_a"])

    health = mgr.namespace_health(namespace="ns")
    assert health["trust"]["demoted"] + health["trust"]["retired"] >= 1
    assert health["avg_confidence"] < 1.0
