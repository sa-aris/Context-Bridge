"""Ontology alignment and the collaboration-quality score."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "onto",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'onto.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


def test_align_merges_surface_variants(tmp_path):
    mgr = _manager(tmp_path)
    graph = mgr.cog.graph
    # Two agents name the same entity differently.
    graph.add_edge(
        namespace="ns", source="service alpha", relation="uses", target="Database-1", memory_id=None
    )
    graph.add_edge(
        namespace="ns",
        source="service alpha",
        relation="reads",
        target="database  1",
        memory_id=None,
    )
    out = mgr.align_graph(namespace="ns")
    assert out["groups_merged"] == 1
    assert out["aliases_created"] == 1

    edges = mgr.graph_neighbors(namespace="ns", entity="service alpha", hops=1)
    targets = {e["target"] for e in edges}
    # Both edges now point at a single canonical name.
    assert len(targets) == 1


def test_alias_resolves_future_edges(tmp_path):
    mgr = _manager(tmp_path)
    assert mgr.add_alias(namespace="ns", alias="db one", canonical="Database One")
    mgr.cog.graph.add_edge(
        namespace="ns", source="api", relation="uses", target="db-one", memory_id=None
    )
    edges = mgr.graph_neighbors(namespace="ns", entity="api", hops=1)
    assert edges[0]["target"] == "Database One"


def test_add_alias_is_noop_when_same(tmp_path):
    mgr = _manager(tmp_path)
    assert mgr.add_alias(namespace="ns", alias="thing", canonical="thing") is False


def test_forget_clears_aliases(tmp_path):
    mgr = _manager(tmp_path)
    mgr.add_alias(namespace="gone", alias="x", canonical="X canonical")
    assert mgr.list_aliases(namespace="gone")
    mgr.forget(namespace="gone")
    assert mgr.list_aliases(namespace="gone") == []


def test_quality_score_reflects_activity(tmp_path):
    mgr = _manager(tmp_path, feedback_weight=0.5)
    res = mgr.write(
        content="the cache layer uses redis", agent_id="a", session_id="s", namespace="ns"
    )
    # A query that hits memory, plus positive feedback.
    mgr.query(query="cache redis", namespace="ns", session_id="s")
    mgr.record_feedback(memory_id=res.ids[0], namespace="ns", useful=True)

    q = mgr.collaboration_quality(namespace="ns")
    assert q["writes"] == 1
    assert q["queries"] == 1
    assert q["hit_rate"] == 1.0
    assert q["feedback_positivity"] == 1.0
    assert q["conflict_health"] == 1.0
    assert q["score"] == 100.0
    assert q["agents"] == 1


def test_quality_empty_namespace_is_neutral(tmp_path):
    mgr = _manager(tmp_path)
    q = mgr.collaboration_quality(namespace="empty")
    assert q["score"] == 20.0  # only conflict_health (1.0 * 0.2) applies with no activity
    assert q["hit_rate"] == 0.0
