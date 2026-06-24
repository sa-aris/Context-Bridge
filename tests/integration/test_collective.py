"""Collective learning: agent reputation, outcome credit, procedural memory."""

from __future__ import annotations

from context_bridge.api.deps import build_container
from context_bridge.config import Settings


def _manager(tmp_path, **overrides):
    base = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "coll",
        "embed_provider": "hashing",
        "embed_dim": 128,
        "rerank_provider": "identity",
        "working_provider": "memory",
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'coll.db'}",
    }
    base.update(overrides)
    return build_container(Settings(**base)).manager


def test_write_builds_agent_profile(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="first finding", agent_id="scout", session_id="s", namespace="ns")
    board = mgr.agent_leaderboard(namespace="ns")
    assert any(a["agent_id"] == "scout" and a["writes"] == 1 for a in board)


def test_feedback_credits_author_reputation(tmp_path):
    mgr = _manager(tmp_path)
    res = mgr.write(
        content="useful fact about caching", agent_id="ace", session_id="s", namespace="ns"
    )
    mgr.record_feedback(memory_id=res.ids[0], namespace="ns", useful=True, weight=2.0)
    board = {a["agent_id"]: a for a in mgr.agent_leaderboard(namespace="ns")}
    assert board["ace"]["useful"] == 1
    assert board["ace"]["score"] > 0


def test_outcome_credits_session_memories_and_agents(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="step one done", agent_id="a1", session_id="run", namespace="ns")
    mgr.write(content="step two done", agent_id="a2", session_id="run", namespace="ns")

    out = mgr.record_outcome(session_id="run", namespace="ns", success=True, weight=1.0)
    assert out["memories_credited"] >= 2
    assert out["agents_credited"] == 2
    assert all(a["score"] > 0 for a in mgr.agent_leaderboard(namespace="ns"))


def test_procedure_lifecycle(tmp_path):
    mgr = _manager(tmp_path)
    pid = mgr.create_procedure(
        namespace="ns",
        title="rotate credentials",
        steps=["revoke old key", "issue new key", "update secrets"],
        created_by="ops",
    )
    assert pid
    mgr.record_procedure_outcome(pid, success=True)
    procs = mgr.list_procedures(namespace="ns", query="rotate")
    assert procs and procs[0]["success_count"] == 1
    assert procs[0]["success_rate"] == 1.0


def test_forget_clears_agents_and_procedures(tmp_path):
    mgr = _manager(tmp_path)
    mgr.write(content="x", agent_id="a", session_id="s", namespace="gone")
    mgr.create_procedure(namespace="gone", title="p", steps=["a"])
    mgr.forget(namespace="gone")
    assert mgr.agent_leaderboard(namespace="gone") == []
    assert mgr.list_procedures(namespace="gone") == []
