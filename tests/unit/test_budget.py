from __future__ import annotations

from context_bridge.core.models import Provenance, RetrievedChunk
from context_bridge.core.retrieval.budget import assemble
from context_bridge.tokenizer import count_tokens


def _chunk(cid: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=content,
        score=1.0,
        namespace="default",
        provenance=Provenance(agent_id="a", session_id="s"),
        parent_id="p",
        parent_text=f"PARENT[{content}]",
    )


def test_budget_is_respected():
    chunks = [_chunk(str(i), " ".join(["token"] * 50)) for i in range(10)]
    result = assemble(chunks, token_budget=60)

    assert result.tokens_used <= 60
    assert 0 < len(result.chunks) < 10


def test_assembly_preserves_order_and_sources():
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta")]
    result = assemble(chunks, token_budget=1000)

    assert "alpha" in result.context
    assert "beta" in result.context
    assert [s["id"] for s in result.sources] == ["a", "b"]


def test_expand_parents_uses_parent_text():
    chunks = [_chunk("a", "alpha")]
    result = assemble(chunks, token_budget=1000, expand_parents=True)
    assert "PARENT[alpha]" in result.context


def test_zero_budget_returns_empty():
    result = assemble([_chunk("a", "alpha")], token_budget=0)
    assert result.context == ""
    assert result.chunks == []
    assert count_tokens(result.context) == 0
