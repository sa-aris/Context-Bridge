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
from context_bridge.core.graph.extractor import Extractor, RuleBasedExtractor
from context_bridge.core.memory.consolidation import cluster_by_similarity
from context_bridge.core.memory.contradiction import ContradictionDetector, NullDetector
from context_bridge.core.memory.policy import WritePolicy, cosine
from context_bridge.core.memory.redaction import NullRedactor, Redactor
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
from context_bridge.core.tracing import span
from context_bridge.core.vectorstore.base import VectorStore
from context_bridge.core.working.base import WorkingMemory
from context_bridge.db.repository import (
    AgentProfileRepository,
    ConflictRepository,
    EpisodeRepository,
    FeedbackRepository,
    GraphRepository,
    ParentRepository,
    ProcedureRepository,
)

_EPISODE_CONTENT_CAP = 2000


@dataclass(slots=True)
class CognitiveServices:
    """Optional, opt-in cognitive-layer components used by the manager."""

    redactor: Redactor = field(default_factory=NullRedactor)
    detector: ContradictionDetector = field(default_factory=NullDetector)
    extractor: Extractor = field(default_factory=RuleBasedExtractor)
    feedback: FeedbackRepository | None = None
    conflicts: ConflictRepository | None = None
    graph: GraphRepository | None = None
    agents: AgentProfileRepository | None = None
    procedures: ProcedureRepository | None = None
    graph_extraction: bool = False
    contradiction_similarity: float = 0.8


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
        cognitive: CognitiveServices | None = None,
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
        self.cog = cognitive or CognitiveServices()

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

        content = self.cog.redactor.redact(content)

        if summarize_before_store:
            content = self.summarizer.summarize(content)

        chunks = self.chunker.chunk(content, parent_id=parent_id)
        if not chunks:
            return WriteResult()

        texts = [c.text for c in chunks]
        with span("memory.embed", chunks=len(texts)):
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
            with span("memory.upsert", records=len(records)):
                self.store.upsert(records)
                self._persist_parents(chunks, records, namespace)
            self._detect_conflicts(records, namespace)
            self._extract_graph(content, namespace, records[0].id)
            if self.cog.agents is not None:
                self.cog.agents.record_write(namespace=namespace, agent_id=agent_id)

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

    def _detect_conflicts(self, records: list[MemoryRecord], namespace: str) -> None:
        """Flag near-duplicate-but-disagreeing memories as conflicts."""
        if self.cog.conflicts is None:
            return
        for rec in records:
            if rec.dense is None:
                continue
            neighbors = self.store.hybrid_search(
                dense=rec.dense, sparse=rec.sparse, limit=3, namespace=namespace
            )
            for nb in neighbors:
                if nb.id == rec.id or nb.dense is None:
                    continue
                sim = cosine(rec.dense, nb.dense)
                if sim >= self.cog.contradiction_similarity and self.cog.detector.is_contradiction(
                    rec.content, nb.content
                ):
                    self.cog.conflicts.add(
                        namespace=namespace,
                        memory_id_a=rec.id,
                        memory_id_b=nb.id,
                        similarity=sim,
                    )
                    break

    def _extract_graph(self, content: str, namespace: str, memory_id: str) -> None:
        """Extract entity/relation triples into the knowledge graph."""
        if not self.cog.graph_extraction or self.cog.graph is None:
            return
        for triple in self.cog.extractor.extract(content):
            self.cog.graph.add_edge(
                namespace=namespace,
                source=triple.source,
                relation=triple.relation,
                target=triple.target,
                memory_id=memory_id,
            )

    # -- cognitive layer --------------------------------------------------
    def record_feedback(
        self, *, memory_id: str, namespace: str, useful: bool, weight: float = 1.0
    ) -> None:
        """Record outcome feedback that re-ranks future recall and credits the author."""
        if self.cog.feedback is None:
            return
        delta = weight if useful else -weight
        self.cog.feedback.record(memory_id=memory_id, namespace=namespace, delta=delta)
        if self.cog.agents is not None:
            chunk = self.store.get(memory_id)
            if chunk is not None:
                self.cog.agents.record_outcome(
                    namespace=namespace,
                    agent_id=chunk.provenance.agent_id,
                    delta=delta,
                    useful=useful,
                )

    def record_outcome(
        self, *, session_id: str, namespace: str, success: bool, weight: float = 1.0
    ) -> dict:
        """Propagate a task outcome as credit to every memory and agent in the session.

        This closes the learning loop: a successful run reinforces the memories it
        produced and the agents that produced them; a failed run does the opposite.
        """
        delta = weight if success else -weight
        memories = 0
        agents: set[str] = set()
        for episode in self.episodes.timeline(session_id):
            if episode["kind"] != "write":
                continue
            for chunk_id in episode["chunk_ids"]:
                if self.cog.feedback is not None:
                    self.cog.feedback.record(memory_id=chunk_id, namespace=namespace, delta=delta)
                    memories += 1
            agent_id = episode["agent_id"]
            if self.cog.agents is not None and agent_id:
                self.cog.agents.record_outcome(
                    namespace=namespace, agent_id=agent_id, delta=delta, useful=success
                )
                agents.add(agent_id)
        return {"memories_credited": memories, "agents_credited": len(agents), "success": success}

    def agent_leaderboard(self, *, namespace: str, limit: int = 20) -> list[dict]:
        return self.cog.agents.top(namespace=namespace, limit=limit) if self.cog.agents else []

    def create_procedure(
        self,
        *,
        namespace: str,
        title: str,
        steps: list[str],
        tags: list[str] | None = None,
        created_by: str | None = None,
    ) -> str | None:
        if self.cog.procedures is None:
            return None
        return self.cog.procedures.create(
            namespace=namespace, title=title, steps=steps, tags=tags, created_by=created_by
        )

    def list_procedures(
        self, *, namespace: str, query: str | None = None, limit: int = 50
    ) -> list[dict]:
        return (
            self.cog.procedures.list(namespace=namespace, query=query, limit=limit)
            if self.cog.procedures
            else []
        )

    def record_procedure_outcome(self, procedure_id: str, *, success: bool) -> bool:
        if self.cog.procedures is None:
            return False
        return self.cog.procedures.record_outcome(procedure_id, success=success)

    def consolidate(
        self,
        *,
        namespace: str,
        min_cluster: int,
        similarity: float,
        max_items: int = 500,
        max_sentences: int = 5,
        agent_id: str = "consolidator",
    ) -> dict:
        """Cluster a namespace's memories and write back synthesized insights."""
        listed, _ = self.store.list_records(namespace=namespace, limit=max_items, cursor=None)
        # Consolidate only raw memories, never previously-synthesized insights.
        chunks = [c for c in listed if c.provenance.source != "consolidation"]
        if len(chunks) < min_cluster:
            return {"scanned": len(chunks), "clusters": 0, "insights": 0}

        vectors = self.embedder.embed_dense([c.content for c in chunks])
        groups = cluster_by_similarity(vectors, threshold=similarity)

        insights = 0
        used_clusters = 0
        for group in groups:
            if len(group) < min_cluster:
                continue
            used_clusters += 1
            merged = "\n".join(chunks[i].content for i in group)
            summary = self.summarizer.summarize(merged, max_sentences=max_sentences)
            if not summary.strip():
                continue
            # Insights are intentionally new artifacts, so they bypass dedup.
            result = self.write(
                content=summary,
                agent_id=agent_id,
                session_id="consolidation",
                namespace=namespace,
                source="consolidation",
                tags=["insight"],
                dedup=False,
            )
            insights += result.stored
        return {"scanned": len(chunks), "clusters": used_clusters, "insights": insights}

    def graph_neighbors(self, *, namespace: str, entity: str, hops: int = 1) -> list[dict]:
        if self.cog.graph is None:
            return []
        return self.cog.graph.neighbors(namespace=namespace, entity=entity, hops=hops)

    def list_conflicts(self, *, namespace: str | None = None, status: str | None = None) -> list:
        if self.cog.conflicts is None:
            return []
        return self.cog.conflicts.list(namespace=namespace, status=status)

    def resolve_conflict(self, conflict_id: str, *, winner_id: str | None) -> bool:
        if self.cog.conflicts is None:
            return False
        return self.cog.conflicts.resolve(conflict_id, winner_id=winner_id)

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
        with span("memory.query", namespace=namespace, top_k=params.top_k):
            result = self.retriever.retrieve(
                query, namespace=namespace, filters=filters, params=params
            )
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

    def forget(self, *, namespace: str | None = None, session_id: str | None = None) -> dict:
        """Erase all memory for a namespace and/or session (right-to-be-forgotten)."""
        if not namespace and not session_id:
            raise ValueError("forget requires a namespace and/or session_id")
        vectors = self.store.delete_by(namespace=namespace, session_id=session_id)
        episodes = self.episodes.delete_by(namespace=namespace, session_id=session_id)
        parents = self.parents.delete_by_namespace(namespace) if namespace else 0
        if namespace and self.cog.graph is not None:
            self.cog.graph.delete_by_namespace(namespace)
        if namespace and self.cog.agents is not None:
            self.cog.agents.delete_by_namespace(namespace)
        if namespace and self.cog.procedures is not None:
            self.cog.procedures.delete_by_namespace(namespace)
        return {
            "vectors_deleted": vectors,
            "episodes_deleted": episodes,
            "parents_deleted": parents,
        }

    # -- working memory ---------------------------------------------------
    def remember_turn(self, session_id: str, item: dict) -> None:
        self.working.append(session_id, item)

    def recent_turns(self, session_id: str, limit: int = 20) -> list[dict]:
        return self.working.recent(session_id, limit=limit)
