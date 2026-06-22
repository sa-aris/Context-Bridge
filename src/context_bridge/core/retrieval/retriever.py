"""The end-to-end retrieval pipeline.

Stages: embed query -> hybrid search (dense + sparse, RRF) -> drop expired ->
cross-encoder rerank -> MMR diversification -> token-budgeted assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.models import AssembledContext, RetrievedChunk, now_ts
from context_bridge.core.retrieval.budget import assemble
from context_bridge.core.retrieval.mmr import mmr_select
from context_bridge.core.retrieval.reranker import Reranker
from context_bridge.core.vectorstore.base import VectorStore


@dataclass(slots=True)
class RetrievalParams:
    """Per-query knobs with sensible, overridable defaults."""

    top_k: int = 8
    token_budget: int = 2048
    candidate_pool: int = 50
    mmr_lambda: float = 0.6
    rerank: bool = True
    expand_parents: bool = False


class Retriever:
    """Composes the vector store, reranker and budgeting into one call."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        filters: dict | None = None,
        params: RetrievalParams,
    ) -> AssembledContext:
        dense = self.embedder.embed_query_dense(query)
        sparse = (
            self.embedder.embed_query_sparse(query)
            if self.embedder.supports_sparse
            else None
        )

        candidates = self.store.hybrid_search(
            dense=dense,
            sparse=sparse,
            limit=max(params.candidate_pool, params.top_k),
            namespace=namespace,
            filters=filters,
        )
        candidates = self._drop_expired(candidates)
        if not candidates:
            return AssembledContext(context="", chunks=[], tokens_used=0)

        if params.rerank:
            candidates = self.reranker.rerank(query, candidates)

        selected = mmr_select(candidates, lambda_=params.mmr_lambda, top_k=params.top_k)
        return assemble(
            selected,
            token_budget=params.token_budget,
            expand_parents=params.expand_parents,
        )

    @staticmethod
    def _drop_expired(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        at = now_ts()
        return [c for c in chunks if not c.provenance.is_expired(at=at)]
