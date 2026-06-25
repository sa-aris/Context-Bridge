"""Belief revision, auto lesson distillation, and recall explainability."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "belief",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'belief.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


# -- belief revision --------------------------------------------------------
def test_resolving_conflict_decays_loser_confidence(tmp_path):
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

    conflict = mgr.list_conflicts(namespace="ns")[0]
    winner, loser = conflict["memory_id_a"], conflict["memory_id_b"]

    assert mgr.resolve_conflict(conflict["id"], winner_id=winner) is True
    assert mgr.get(loser).provenance.confidence < 1.0
    # The winner keeps full trust.
    assert mgr.get(winner).provenance.confidence == 1.0


def test_resolve_without_winner_keeps_confidence(tmp_path):
    mgr = _manager(
        tmp_path, detect_contradictions=True, contradiction_similarity=0.5, dedup_threshold=0.99
    )
    mgr.write(content="the cache is warm", agent_id="a", session_id="s", namespace="ns")
    mgr.write(content="the cache is not warm", agent_id="b", session_id="s", namespace="ns")
    conflict = mgr.list_conflicts(namespace="ns")[0]
    assert mgr.resolve_conflict(conflict["id"], winner_id=None) is True
    assert mgr.get(conflict["memory_id_b"]).provenance.confidence == 1.0


def test_decayed_memory_is_demoted_in_recall(tmp_path):
    mgr = _manager(
        tmp_path,
        detect_contradictions=True,
        contradiction_similarity=0.5,
        dedup_threshold=0.99,
        confidence_weight=0.9,
    )
    mgr.write(content="the scheduler runs hourly", agent_id="a", session_id="s", namespace="ns")
    mgr.write(
        content="the scheduler does not run hourly", agent_id="b", session_id="s", namespace="ns"
    )
    conflict = mgr.list_conflicts(namespace="ns")[0]
    loser = conflict["memory_id_b"]

    before = {c.id: c.score for c in mgr.query(query="scheduler hourly", namespace="ns").chunks}
    mgr.resolve_conflict(conflict["id"], winner_id=conflict["memory_id_a"])
    after = {c.id: c for c in mgr.query(query="scheduler hourly", namespace="ns").chunks}

    # Belief revision lowers the discredited memory's recall score...
    assert after[loser].score < before[loser]
    # ...and recall explains why it was demoted.
    assert after[loser].signals["confidence"] < 1.0


# -- auto lesson distillation ----------------------------------------------
def test_distill_lessons_from_repeated_failures(tmp_path):
    mgr = _manager(tmp_path)
    ids: list[str] = []
    for _ in range(3):
        res = mgr.write(
            content="the deploy script skipped the database migration step",
            agent_id="a",
            session_id="s",
            namespace="ns",
            dedup=False,
        )
        ids += res.ids
    for mid in ids:
        mgr.record_feedback(memory_id=mid, namespace="ns", useful=False, weight=2.0)

    out = mgr.distill_lessons(namespace="ns", min_cluster=2, similarity=0.5)
    assert out["scanned"] >= 3
    assert out["lessons_created"] >= 1
    assert mgr.list_lessons(namespace="ns")


def test_distill_is_noop_without_failures(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="a perfectly fine memory", agent_id="a", session_id="s", namespace="ns")
    out = mgr.distill_lessons(namespace="ns", min_cluster=2, similarity=0.5)
    assert out == {"scanned": 0, "clusters": 0, "lessons_created": 0}


# -- explainability ---------------------------------------------------------
def test_recall_results_carry_signals(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="redis powers the cache layer", agent_id="a", session_id="s", namespace="ns")
    chunk = mgr.query(query="cache redis", namespace="ns").chunks[0]
    assert "match" in chunk.signals
    assert "age_days" in chunk.signals


def test_feedback_shows_up_as_signal(tmp_path):
    mgr = _manager(tmp_path, feedback_weight=0.5)
    res = mgr.write(
        content="the billing job runs nightly", agent_id="a", session_id="s", namespace="ns"
    )
    mgr.record_feedback(memory_id=res.ids[0], namespace="ns", useful=True, weight=5)
    chunk = mgr.query(query="billing nightly", namespace="ns").chunks[0]
    assert chunk.signals.get("feedback", 0) > 0
