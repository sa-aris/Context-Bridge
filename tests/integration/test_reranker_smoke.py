"""Smoke test for the real FastEmbed cross-encoder reranker.

Skips automatically when ``fastembed`` is not installed or the model cannot be
fetched (e.g. offline CI), so the default hermetic suite stays green either way.
"""

from __future__ import annotations

import pytest

from context_bridge.core.models import Provenance, RetrievedChunk

pytest.importorskip("fastembed")


def _chunk(cid: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=content,
        score=score,
        namespace="default",
        provenance=Provenance(agent_id="a", session_id="s"),
        parent_id="p",
    )


@pytest.fixture(scope="module")
def reranker():
    from context_bridge.core.retrieval.reranker import FastEmbedReranker

    rr = FastEmbedReranker()
    try:
        rr.rerank("warmup", [_chunk("w", "warmup text", 0.0)])
    except Exception as exc:  # pragma: no cover - network/model unavailable
        pytest.skip(f"fastembed model unavailable: {exc}")
    return rr


def test_cross_encoder_promotes_relevant_chunk(reranker):
    # Fusion order is intentionally "wrong" so a no-op reranker would fail.
    chunks = [
        _chunk("coffee", "The office coffee machine is broken.", score=0.9),
        _chunk("pay", "The payment service retries failed charges three times.", score=0.1),
    ]
    reranked = reranker.rerank("how are failed payments handled?", chunks)

    assert reranked[0].id == "pay"
    assert reranked[0].score > reranked[1].score
