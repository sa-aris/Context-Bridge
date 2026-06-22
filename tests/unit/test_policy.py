from __future__ import annotations

from context_bridge.core.embeddings.hashing import HashingEmbedder
from context_bridge.core.memory.policy import WritePolicy, cosine
from context_bridge.core.models import Provenance, RetrievedChunk


def _neighbor(dense: list[float]) -> RetrievedChunk:
    return RetrievedChunk(
        id="n",
        content="neighbor",
        score=1.0,
        namespace="default",
        provenance=Provenance(agent_id="a", session_id="s"),
        parent_id="p",
        dense=dense,
    )


def test_identical_text_is_duplicate():
    embedder = HashingEmbedder(dim=128)
    vec = embedder.embed_query_dense("the build pipeline failed on step three")
    policy = WritePolicy(dedup_threshold=0.95)
    assert policy.is_duplicate(vec, _neighbor(vec)) is True


def test_distinct_text_is_not_duplicate():
    embedder = HashingEmbedder(dim=128)
    a = embedder.embed_query_dense("the build pipeline failed on step three")
    b = embedder.embed_query_dense("lunch options near the office are limited")
    policy = WritePolicy(dedup_threshold=0.95)
    assert policy.is_duplicate(a, _neighbor(b)) is False


def test_confidence_gate():
    policy = WritePolicy(min_confidence=0.5)
    assert policy.passes_confidence(0.6) is True
    assert policy.passes_confidence(0.4) is False


def test_cosine_bounds():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
