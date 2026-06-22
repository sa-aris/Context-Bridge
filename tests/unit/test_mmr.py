from __future__ import annotations

from context_bridge.core.models import Provenance, RetrievedChunk
from context_bridge.core.retrieval.mmr import mmr_select


def _chunk(cid: str, score: float, dense: list[float]) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=cid,
        score=score,
        namespace="default",
        provenance=Provenance(agent_id="a", session_id="s"),
        parent_id="p",
        dense=dense,
    )


def test_mmr_prefers_diverse_over_redundant():
    # Two near-identical high scorers and one distinct, slightly lower scorer.
    a = _chunk("a", 1.0, [1.0, 0.0])
    a2 = _chunk("a2", 0.99, [0.99, 0.01])
    b = _chunk("b", 0.8, [0.0, 1.0])

    selected = mmr_select([a, a2, b], lambda_=0.5, top_k=2)
    ids = [c.id for c in selected]

    assert ids[0] == "a"
    assert "b" in ids  # diversity beats the redundant near-duplicate a2


def test_mmr_returns_all_when_fewer_than_top_k():
    chunks = [_chunk("a", 1.0, [1.0, 0.0]), _chunk("b", 0.5, [0.0, 1.0])]
    assert len(mmr_select(chunks, top_k=5)) == 2


def test_mmr_handles_empty():
    assert mmr_select([], top_k=5) == []
