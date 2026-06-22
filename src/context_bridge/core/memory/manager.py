"""The single orchestration facade used by both the HTTP API and the SDK.

``MemoryManager`` wires the write path (chunk -> embed -> govern -> persist ->
log) and the read path (delegated to :class:`Retriever`) together, plus session
summarisation and working-memory access. Every public method is intentionally
side-effect-explicit so callers can reason about cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from context_bridge.core.chunking.base import Chunker
from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.memory.policy import WritePolicy, cosine
from context_bridge.core.memory.summarizer import ExtractiveSummarizer, Summarizer
from context_bridge.core.models import (
    AssembledContext,
    Chunk,
    MemoryRecord,
    Provenance,
    SparseVector,
    now_ts,
)
from context_bridge.core.retrieval.retriever import RetrievalParams, Retriever
from context_bridge.core.vectorstore.base import VectorStore
from context_bridge.core.working.base import WorkingMemory
from context_bridge.db.repository import EpisodeRepository, ParentRepository

_EPISODE_CONTENT_CAP = 2000


@dataclass(slots=True)
class WriteResult:
    """Outcome of a write: which chunks were stored vs. suppressed."""

    ids: list[str] = field(default_factory=list)
    stored: int = 0
    deduped: int = 0
    skipped: bool = False


class MemoryManager:
    """Coordinates chunking, embedding, governance, persistence and retrieval."""

    def __init__(
        self,
        *,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        retriever: Retriever,
        working: WorkingMemory,
        episodes: EpisodeRepository,
        parents: ParentRepository,
        policy: WritePolicy,
        defaults: RetrievalParams,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.working = working
        self.episodes = episodes
        self.parents = parents
        self.policy = policy
        self.defaults = defaults
        self.summarizer = summarizer or ExtractiveSummarizer()

    # -- write path -------------------------------------------------------
    def write(
        self,
        *,
        content: str,
        agent_id: str,
        session_id: str,
        task_id: str | None = None,
        namespace: str = "default",
        tags: list[str] | None = None,
        confidence: float = 1.0,
        metadata: dict | None = None,
        source: str | None = None,
        ttl_seconds: int | None = None,
        dedup: bool = True,
        summarize_before_store: bool = False,
        parent_id: str | None = None,
    ) -> WriteResult:
        if not self.policy.passes_confidence(confidence):
            return WriteResult(skipped=True)

        if summarize_before_store:
            content = self.summarizer.summarize(content)

        chunks = self.chunker.chunk(content, parent_id=parent_id)
        if not chunks:
            return WriteResult()

        texts = [c.text for c in chunks]
        dense_vecs = self.embedder.embed_dense(texts)
        sparse_vecs: list[SparseVector | None]
        if self.embedder.supports_sparse:
            sparse_vecs = list(self.embedder.embed_sparse(texts))
        else:
            sparse_vecs = [None] * len(texts)

        records: list[MemoryRecord] = []
        deduped = 0
        for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True):
            if dedup and self._is_duplicate(dense, sparse, namespace, records):
                deduped += 1
                continue
            records.append(
                MemoryRecord(
                    id=chunk.id,
                    content=chunk.text,
                    namespace=namespace,
                    provenance=Provenance(
                        agent_id=agent_id,
                        session_id=session_id,
                        task_id=task_id,
                        source=source,
                        confidence=confidence,
                        created_at=now_ts(),
                        ttl_seconds=ttl_seconds,
                    ),
                    parent_id=chunk.parent_id,
                    # parent text lives once in the parent store, not in the payload
                    parent_text=None,
                    tags=tags or [],
                    metadata=metadata or {},
                    dense=dense,
                    sparse=sparse,
                )
            )

        if records:
            self.store.upsert(records)
            self._persist_parents(chunks, records, namespace)

        ids = [r.id for r in records]
        self.episodes.record(
            session_id=session_id,
            agent_id=agent_id,
            task_id=task_id,
            namespace=namespace,
            kind="write",
            content=content[:_EPISODE_CONTENT_CAP],
            chunk_ids=ids,
            confidence=confidence,
        )
        self.working.append(
            session_id,
            {"kind": "write", "agent_id": agent_id, "content": content[:500]},
        )
        return WriteResult(ids=ids, stored=len(ids), deduped=deduped)

    def _persist_parents(
        self, chunks: list[Chunk], records: list[MemoryRecord], namespace: str
    ) -> None:
        """Store each stored chunk's parent document once (small-to-big)."""
        stored_parents = {r.parent_id for r in records}
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.parent_id in stored_parents and chunk.parent_id not in seen:
                seen.add(chunk.parent_id)
                self.parents.upsert(
                    parent_id=chunk.parent_id,
                    namespace=namespace,
                    text=chunk.parent_text or chunk.text,
                )

    def _is_duplicate(
        self,
        dense: list[float],
        sparse: SparseVector | None,
        namespace: str,
        pending: list[MemoryRecord],
    ) -> bool:
        # Suppress duplicates already accepted in this same batch...
        for rec in pending:
            if rec.dense is not None and cosine(dense, rec.dense) >= self.policy.dedup_threshold:
                return True
        # ...and near-duplicates already living in the store.
        neighbors = self.store.hybrid_search(
            dense=dense, sparse=sparse, limit=1, namespace=namespace
        )
        neighbor = neighbors[0] if neighbors else None
        return self.policy.is_duplicate(dense, neighbor)

    # -- read path --------------------------------------------------------
    def query(
        self,
        *,
        query: str,
        namespace: str = "default",
        agent_id: str = "system",
        session_id: str | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
        filters: dict | None = None,
        rerank: bool = True,
        expand_parents: bool = False,
    ) -> AssembledContext:
        params = RetrievalParams(
            top_k=top_k or self.defaults.top_k,
            token_budget=token_budget or self.defaults.token_budget,
            candidate_pool=self.defaults.candidate_pool,
            mmr_lambda=self.defaults.mmr_lambda,
            rerank=rerank,
            expand_parents=expand_parents,
        )
        result = self.retriever.retrieve(query, namespace=namespace, filters=filters, params=params)
        if session_id:
            self.episodes.record(
                session_id=session_id,
                agent_id=agent_id,
                namespace=namespace,
                kind="query",
                content=query[:_EPISODE_CONTENT_CAP],
                chunk_ids=[c.id for c in result.chunks],
            )
        return result

    # -- maintenance ------------------------------------------------------
    def summarize_session(
        self,
        *,
        session_id: str,
        namespace: str = "default",
        agent_id: str = "summarizer",
        max_sentences: int = 5,
        store_summary: bool = True,
    ) -> dict:
        """Compress a session's writes into a single summary memory."""
        timeline = self.episodes.timeline(session_id)
        source_text = "\n".join(
            e["content"] for e in timeline if e["kind"] == "write" and e["content"]
        )
        summary = self.summarizer.summarize(source_text, max_sentences=max_sentences)

        stored_ids: list[str] = []
        if summary and store_summary:
            result = self.write(
                content=summary,
                agent_id=agent_id,
                session_id=session_id,
                namespace=namespace,
                source="session-summary",
                tags=["summary"],
                dedup=False,
            )
            stored_ids = result.ids
        return {"summary": summary, "chunk_ids": stored_ids}

    def timeline(self, session_id: str, *, limit: int = 100) -> list[dict]:
        return self.episodes.timeline(session_id, limit=limit)

    def get(self, record_id: str):
        return self.store.get(record_id)

    def delete(self, record_ids: list[str]) -> None:
        self.store.delete(record_ids)

    def sweep_expired(self) -> int:
        """Physically remove TTL-expired memories from the semantic store."""
        return self.store.sweep_expired()

    def list_records(self, *, namespace: str | None, limit: int, cursor: str | None):
        """Page through stored memories, optionally filtered by namespace."""
        return self.store.list_records(namespace=namespace, limit=limit, cursor=cursor)

    # -- working memory ---------------------------------------------------
    def remember_turn(self, session_id: str, item: dict) -> None:
        self.working.append(session_id, item)

    def recent_turns(self, session_id: str, limit: int = 20) -> list[dict]:
        return self.working.recent(session_id, limit=limit)
