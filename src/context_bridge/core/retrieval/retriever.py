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
FeedbackLookup = Callable[[list[str]], dict[str, float]]


@dataclass(slots=True)
class RetrievalParams:
    """Per-query knobs with sensible, overridable defaults."""

    top_k: int = 8
    token_budget: int = 2048
    candidate_pool: int = 50
    mmr_lambda: float = 0.6
    rerank: bool = True
    expand_parents: bool = False
    include_dates: bool = False
    since: float | None = None
    until: float | None = None


class Retriever:
    """Composes the vector store, reranker and budgeting into one call."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        parent_lookup: ParentLookup | None = None,
        feedback_lookup: FeedbackLookup | None = None,
        feedback_weight: float = 0.0,
        confidence_weight: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.parent_lookup = parent_lookup
        self.feedback_lookup = feedback_lookup
        self.feedback_weight = feedback_weight
        self.confidence_weight = confidence_weight

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
        candidates = self._filter_temporal(candidates, params.since, params.until)
        if not candidates:
            return AssembledContext(context="", chunks=[], tokens_used=0)

        if params.rerank:
            with span("retrieve.rerank", candidates=len(candidates)):
                candidates = self.reranker.rerank(query, candidates)

        for chunk in candidates:
            chunk.signals["match"] = round(chunk.score, 4)

        if self.feedback_lookup is not None and self.feedback_weight:
            self._apply_feedback(candidates)
        if self.confidence_weight:
            self._apply_confidence(candidates)
        self._annotate_recency(candidates)

        with span("retrieve.assemble", top_k=params.top_k, token_budget=params.token_budget):
            selected = mmr_select(candidates, lambda_=params.mmr_lambda, top_k=params.top_k)
            if params.expand_parents and self.parent_lookup is not None:
                self._hydrate_parents(selected)
            return assemble(
                selected,
                token_budget=params.token_budget,
                expand_parents=params.expand_parents,
                include_dates=params.include_dates,
            )

    def _apply_feedback(self, chunks: list[RetrievedChunk]) -> None:
        """Nudge candidate scores by accumulated outcome feedback."""
        assert self.feedback_lookup is not None
        scores = self.feedback_lookup([c.id for c in chunks])
        if not scores:
            return
        import math

        for chunk in chunks:
            signal = scores.get(chunk.id)
            if signal:
                adjustment = self.feedback_weight * math.tanh(signal)
                chunk.score += adjustment
                chunk.signals["feedback"] = round(adjustment, 4)

    def _apply_confidence(self, chunks: list[RetrievedChunk]) -> None:
        """Demote low-confidence memories (e.g. losers of a resolved conflict).

        Confidence is blended in multiplicatively, so a memory whose trust has
        decayed sinks in the ranking while fully-trusted ones are untouched.
        """
        w = self.confidence_weight
        for chunk in chunks:
            confidence = chunk.provenance.confidence
            if confidence >= 1.0:
                continue
            factor = (1.0 - w) + w * max(confidence, 0.0)
            chunk.score *= factor
            chunk.signals["confidence"] = round(factor, 4)

    @staticmethod
    def _annotate_recency(chunks: list[RetrievedChunk]) -> None:
        now = now_ts()
        for chunk in chunks:
            ts = chunk.provenance.created_at
            if ts:
                chunk.signals["age_days"] = round((now - ts) / 86400.0, 1)

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

    @staticmethod
    def _filter_temporal(
        chunks: list[RetrievedChunk], since: float | None, until: float | None
    ) -> list[RetrievedChunk]:
        if since is None and until is None:
            return chunks
        out = []
        for c in chunks:
            ts = c.provenance.created_at
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            out.append(c)
        return out
