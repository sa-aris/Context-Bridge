from __future__ import annotations

import pytest

from context_bridge.config import Settings
from context_bridge.core.embeddings import build_embedder
from context_bridge.core.embeddings.cohere_embedder import CohereEmbedder
from context_bridge.core.embeddings.openai_embedder import OpenAIEmbedder
from context_bridge.core.retrieval.reranker import CohereReranker, build_reranker


def test_build_openai_embedder_dispatch_and_dim():
    emb = build_embedder(
        Settings(embed_provider="openai", embed_dense_model="text-embedding-3-large")
    )
    assert isinstance(emb, OpenAIEmbedder)
    assert emb.dense_dim == 3072
    assert emb.supports_sparse is False


def test_build_cohere_embedder_dispatch_and_dim():
    emb = build_embedder(Settings(embed_provider="cohere", embed_dense_model="embed-english-v3.0"))
    assert isinstance(emb, CohereEmbedder)
    assert emb.dense_dim == 1024
    assert emb.supports_sparse is False


def test_openai_unknown_model_falls_back_to_configured_dim():
    emb = OpenAIEmbedder(model="some-custom-model", dim=768)
    assert emb.dense_dim == 768


def test_build_cohere_reranker_dispatch():
    rr = build_reranker(Settings(rerank_provider="cohere"))
    assert isinstance(rr, CohereReranker)


def test_unknown_providers_raise():
    with pytest.raises(ValueError):
        build_embedder(Settings(embed_provider="nope"))
    with pytest.raises(ValueError):
        build_reranker(Settings(rerank_provider="nope"))
