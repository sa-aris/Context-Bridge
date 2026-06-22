"""The end-to-end retrieval pipeline.

Stages: embed query -> hybrid search (dense + sparse, RRF) -> drop expired ->
cross-encoder rerank -> MMR diversification -> token-budgeted assembly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.models import AssembledContext, RetrievedChunk, now_ts
from context_bridge.core.retrieval.budget import assemble
from context_bridge.core.retrieval.mmr import mmr_select
from context_bridge.core.retrieval.reranker import Reranker
from context_bridge.core.tracing import span
from context_bridge.core.vectorstore.base import VectorStore

ParentLookup = Callable[[list[str]], dict[str, str]]


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
        parent_lookup: ParentLookup | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.parent_lookup = parent_lookup

    def retrieve(
        self,
        query: str,
        *,
        namespace: str | None = None,
        filters: dict | None = None,
        params: RetrievalParams,
    ) -> AssembledContext:
        with span("retrieve.embed"):
            dense = self.embedder.embed_query_dense(query)
            sparse = (
                self.embedder.embed_query_sparse(query) if self.embedder.supports_sparse else None
            )

        with span("retrieve.search", candidate_pool=params.candidate_pool):
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
            with span("retrieve.rerank", candidates=len(candidates)):
                candidates = self.reranker.rerank(query, candidates)

        with span("retrieve.assemble", top_k=params.top_k, token_budget=params.token_budget):
            selected = mmr_select(candidates, lambda_=params.mmr_lambda, top_k=params.top_k)
            if params.expand_parents and self.parent_lookup is not None:
                self._hydrate_parents(selected)
            return assemble(
                selected,
                token_budget=params.token_budget,
                expand_parents=params.expand_parents,
            )

    def _hydrate_parents(self, chunks: list[RetrievedChunk]) -> None:
        assert self.parent_lookup is not None
        parent_ids = [c.parent_id for c in chunks if c.parent_id]
        texts = self.parent_lookup(parent_ids)
        for chunk in chunks:
            chunk.parent_text = texts.get(chunk.parent_id)

    @staticmethod
    def _drop_expired(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        at = now_ts()
        return [c for c in chunks if not c.provenance.is_expired(at=at)]
