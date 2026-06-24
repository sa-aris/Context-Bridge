"""End-to-end cognitive layer: redaction, feedback, consolidation, conflicts, graph."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "cog",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'cog.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


def test_pii_is_redacted_before_storage(tmp_path):
    mgr = _manager(tmp_path, redact_pii=True)
    mgr.write(
        content="contact jane@example.com about the invoice",
        agent_id="a",
        session_id="s",
        namespace="ns",
    )
    result = mgr.query(query="contact invoice", namespace="ns")
    assert "jane@example.com" not in result.context
    assert "REDACTED" in result.context


def test_feedback_boosts_score(tmp_path):
    mgr = _manager(tmp_path, feedback_weight=0.5)
    res = mgr.write(
        content="the cache layer uses redis", agent_id="a", session_id="s", namespace="ns"
    )
    memory_id = res.ids[0]

    before = mgr.query(query="cache redis", namespace="ns").chunks[0].score
    for _ in range(3):
        mgr.record_feedback(memory_id=memory_id, namespace="ns", useful=True, weight=5)
    after = mgr.query(query="cache redis", namespace="ns").chunks[0].score
    assert after > before


def test_consolidation_creates_insight(tmp_path):
    mgr = _manager(tmp_path)
    for _ in range(3):
        mgr.write(
            content="the billing service charges customers monthly via stripe",
            agent_id="a",
            session_id="s",
            namespace="ns",
            dedup=False,
        )
    out = mgr.consolidate(namespace="ns", min_cluster=2, similarity=0.5)
    assert out["scanned"] >= 3
    assert out["insights"] >= 1


def test_contradiction_is_recorded(tmp_path):
    mgr = _manager(
        tmp_path, detect_contradictions=True, contradiction_similarity=0.5, dedup_threshold=0.99
    )
    mgr.write(
        content="the gateway is enabled in production", agent_id="a", session_id="s", namespace="ns"
    )
    mgr.write(
        content="the gateway is not enabled in production",
        agent_id="b",
        session_id="s",
        namespace="ns",
    )
    conflicts = mgr.list_conflicts(namespace="ns")
    assert len(conflicts) >= 1


def test_graph_extraction_and_neighbors(tmp_path):
    mgr = _manager(tmp_path, graph_extraction=True)
    mgr.write(
        content="service alpha depends on database one. service alpha uses cache two.",
        agent_id="a",
        session_id="s",
        namespace="ns",
    )
    edges = mgr.graph_neighbors(namespace="ns", entity="service alpha", hops=1)
    relations = {(e["source"], e["relation"], e["target"]) for e in edges}
    assert ("service alpha", "depends on", "database one") in relations


def test_forget_clears_graph(tmp_path):
    mgr = _manager(tmp_path, graph_extraction=True)
    mgr.write(
        content="service x depends on service y", agent_id="a", session_id="s", namespace="gone"
    )
    assert mgr.graph_neighbors(namespace="gone", entity="service x")
    mgr.forget(namespace="gone")
    assert mgr.graph_neighbors(namespace="gone", entity="service x") == []
