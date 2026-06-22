"""End-to-end exercise of the write -> govern -> retrieve -> budget pipeline."""

from __future__ import annotations

from context_bridge.core.memory.manager import MemoryManager


def test_write_then_query_returns_relevant_context(manager: MemoryManager):
    manager.write(
        content="The payment service uses Stripe and retries failed charges three times.",
        agent_id="agent-billing",
        session_id="sess-1",
        namespace="proj-x",
    )
    manager.write(
        content="The office coffee machine is broken and a replacement is on order.",
        agent_id="agent-ops",
        session_id="sess-1",
        namespace="proj-x",
    )

    result = manager.query(
        query="how does the payment service handle failed charges?",
        namespace="proj-x",
        session_id="sess-1",
        token_budget=200,
    )

    assert result.chunks, "expected at least one retrieved chunk"
    assert "Stripe" in result.context
    assert result.tokens_used <= 200
    assert result.sources[0]["agent_id"] == "agent-billing"


def test_dedup_suppresses_identical_writes(manager: MemoryManager):
    text = "Deployment freeze is in effect until the end of the quarter."
    first = manager.write(content=text, agent_id="a", session_id="s", namespace="ns")
    second = manager.write(content=text, agent_id="a", session_id="s", namespace="ns")

    assert first.stored == 1
    assert second.stored == 0
    assert second.deduped == 1


def test_namespaces_isolate_memories(manager: MemoryManager):
    manager.write(content="alpha secret lives here", agent_id="a", session_id="s", namespace="A")
    manager.write(content="beta secret lives here", agent_id="a", session_id="s", namespace="B")

    res_a = manager.query(query="secret", namespace="A")
    assert all(c.namespace == "A" for c in res_a.chunks)
    assert any("alpha" in c.content for c in res_a.chunks)


def test_confidence_gate_skips_low_confidence(manager: MemoryManager):
    manager.policy.min_confidence = 0.5
    result = manager.write(
        content="A low confidence guess.", agent_id="a", session_id="s", confidence=0.2
    )
    assert result.skipped is True
    assert result.stored == 0


def test_timeline_records_episodes(manager: MemoryManager):
    manager.write(content="first note", agent_id="a", session_id="time-sess", namespace="ns")
    manager.query(query="note", namespace="ns", session_id="time-sess")

    timeline = manager.timeline("time-sess")
    kinds = [e["kind"] for e in timeline]

    assert "write" in kinds
    assert "query" in kinds


def test_get_and_delete_roundtrip(manager: MemoryManager):
    result = manager.write(content="ephemeral fact", agent_id="a", session_id="s", namespace="ns")
    record_id = result.ids[0]

    assert manager.get(record_id) is not None
    manager.delete([record_id])
    assert manager.get(record_id) is None


def test_expand_parents_returns_broader_context(manager: MemoryManager):
    filler = " ".join(["padding"] * 200)
    content = f"ALPHAMARKER is the start. {filler} OMEGAMARKER is the end."
    manager.write(content=content, agent_id="a", session_id="s", namespace="big")

    narrow = manager.query(query="ALPHAMARKER", namespace="big", token_budget=4096)
    assert "ALPHAMARKER" in narrow.context
    assert "OMEGAMARKER" not in narrow.context  # only the matched small chunk

    expanded = manager.query(
        query="ALPHAMARKER", namespace="big", token_budget=4096, expand_parents=True
    )
    assert "OMEGAMARKER" in expanded.context  # parent document hydrated from SQL


def test_summarize_session_writes_summary(manager: MemoryManager):
    for i in range(6):
        manager.write(
            content=(
                f"Finding {i}: the migration script must run before the API deploy "
                f"because table {i} is referenced by the new endpoints."
            ),
            agent_id="a",
            session_id="sum-sess",
            namespace="ns",
        )

    out = manager.summarize_session(session_id="sum-sess", namespace="ns")
    assert out["summary"]
    assert out["chunk_ids"]
