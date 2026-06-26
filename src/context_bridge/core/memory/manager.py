"""The single orchestration facade used by both the HTTP API and the SDK.

``MemoryManager`` wires the write path (chunk -> embed -> govern -> persist ->
log) and the read path (delegated to :class:`Retriever`) together, plus session
summarisation and working-memory access. Every public method is intentionally
side-effect-explicit so callers can reason about cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from context_bridge.core.chunking.base import Chunker
from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.events import EventEmitter, NullEmitter
from context_bridge.core.graph.extractor import Extractor, RuleBasedExtractor
from context_bridge.core.memory.consolidation import cluster_by_similarity
from context_bridge.core.memory.contradiction import ContradictionDetector, NullDetector
from context_bridge.core.memory.policy import WritePolicy, cosine
from context_bridge.core.memory.redaction import NullRedactor, Redactor
from context_bridge.core.memory.salience import SalienceScorer
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
    LessonRepository,
    ParentRepository,
    ProcedureRepository,
)
from context_bridge.tokenizer import count_tokens

_EPISODE_CONTENT_CAP = 2000

# Ranking nudges so the most actionable lessons surface first.
_SEVERITY_BOOST = {"low": 0.0, "medium": 0.05, "high": 0.1}

# Below this confidence a memory is considered effectively retired (discredited).
_RETIRED_BELOW = 0.1


def _trust_status(confidence: float) -> str:
    """Classify a memory's standing from its confidence."""
    if confidence >= 1.0:
        return "active"
    if confidence < _RETIRED_BELOW:
        return "retired"
    return "demoted"


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
    lessons: LessonRepository | None = None
    graph_extraction: bool = False
    contradiction_similarity: float = 0.8
    lessons_enabled: bool = True
    lessons_top_k: int = 3
    lessons_min_score: float = 0.2
    belief_revision: bool = True
    conflict_loser_decay: float = 0.5


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
        events: EventEmitter | None = None,
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
        self.events = events or NullEmitter()

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
                    self.events.emit(
                        "conflict.opened",
                        namespace,
                        {"memory_id_a": rec.id, "memory_id_b": nb.id, "similarity": round(sim, 4)},
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
        self,
        *,
        session_id: str,
        namespace: str,
        success: bool,
        weight: float = 1.0,
        lesson: str | None = None,
        lesson_trigger: str | None = None,
        severity: str = "medium",
    ) -> dict:
        """Propagate a task outcome as credit to every memory and agent in the session.

        This closes the learning loop: a successful run reinforces the memories it
        produced and the agents that produced them; a failed run does the opposite.
        When a failure carries a ``lesson``, it is captured as durable failure
        memory so the same mistake can be flagged before it recurs.
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

        lesson_id: str | None = None
        if lesson:
            # The trigger defaults to the lesson itself; supply a distinct trigger
            # to describe the *situation* a lesson should fire on.
            lesson_id = self.record_lesson(
                namespace=namespace,
                trigger=lesson_trigger or lesson,
                guidance=lesson,
                severity=severity,
                session_id=session_id,
            )
        return {
            "memories_credited": memories,
            "agents_credited": len(agents),
            "success": success,
            "lesson_id": lesson_id,
        }

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

    def distill_session(
        self,
        *,
        session_id: str,
        namespace: str = "default",
        agent_id: str = "distiller",
        max_promote: int = 5,
        min_score: float = 1.0,
    ) -> dict:
        """Promote the most salient working-memory turns into durable memory.

        This is how a conversation's *dwelled-upon* points survive into future
        sessions: ephemeral turns are scored for salience and only the important
        ones are committed to long-term, cross-session memory.
        """
        turns = self.working.recent(session_id, limit=500)
        texts = [t.get("content", "") for t in turns if isinstance(t, dict)]
        salient = SalienceScorer(min_score=min_score).distill(texts, max_promote=max_promote)
        promoted: list[str] = []
        for item in salient:
            result = self.write(
                content=item.text,
                agent_id=agent_id,
                session_id=session_id,
                namespace=namespace,
                source="distilled",
                tags=["salient"],
                dedup=True,
            )
            promoted.extend(result.ids)
        return {"scanned": len(texts), "promoted": len(promoted), "ids": promoted}

    def graph_neighbors(self, *, namespace: str, entity: str, hops: int = 1) -> list[dict]:
        if self.cog.graph is None:
            return []
        return self.cog.graph.neighbors(namespace=namespace, entity=entity, hops=hops)

    # -- ontology alignment ----------------------------------------------
    def align_graph(self, *, namespace: str) -> dict:
        """Merge surface variants of the same entity onto one canonical name.

        Lets agents that named the same thing differently converge on a shared
        vocabulary, so graph queries no longer fragment across spellings.
        """
        if self.cog.graph is None:
            return {"groups_merged": 0, "aliases_created": 0}
        return self.cog.graph.align(namespace)

    def add_alias(self, *, namespace: str, alias: str, canonical: str) -> bool:
        """Manually declare that ``alias`` refers to ``canonical``."""
        if self.cog.graph is None:
            return False
        return self.cog.graph.register_alias(namespace=namespace, alias=alias, canonical=canonical)

    def list_aliases(self, *, namespace: str) -> list[dict]:
        if self.cog.graph is None:
            return []
        return self.cog.graph.list_aliases(namespace)

    # -- failure memory (learning from past mistakes) --------------------
    def record_lesson(
        self,
        *,
        namespace: str,
        trigger: str,
        guidance: str,
        severity: str = "medium",
        created_by: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Capture a lesson so a past mistake can be flagged before it recurs.

        The ``trigger`` describes the situation the lesson applies to and is
        embedded for semantic matching; the ``guidance`` is what to do or avoid.
        """
        if self.cog.lessons is None:
            return None
        severity = severity if severity in _SEVERITY_BOOST else "medium"
        embedding = self.embedder.embed_query_dense(trigger)
        lesson_id = self.cog.lessons.add(
            namespace=namespace,
            trigger=trigger,
            guidance=guidance,
            embedding=embedding,
            severity=severity,
            created_by=created_by,
            source_session=session_id,
        )
        self.events.emit(
            "lesson.created",
            namespace,
            {"id": lesson_id, "severity": severity, "trigger": trigger[:280]},
        )
        return lesson_id

    def relevant_lessons(
        self,
        *,
        query: str,
        namespace: str,
        top_k: int | None = None,
        min_score: float | None = None,
        reinforce: bool = True,
    ) -> list[dict]:
        """Rank stored lessons by how well they match the situation in ``query``.

        Scoring blends trigger↔query similarity with a small severity and
        proven-helpfulness boost, so the most actionable warnings rise first.
        """
        if self.cog.lessons is None:
            return []
        top_k = top_k if top_k is not None else self.cog.lessons_top_k
        min_score = min_score if min_score is not None else self.cog.lessons_min_score
        stored = self.cog.lessons.with_embeddings(namespace)
        if not stored:
            return []

        q = self.embedder.embed_query_dense(query)
        scored: list[dict] = []
        for lesson in stored:
            embedding = lesson.pop("embedding", [])
            if not embedding:
                continue
            similarity = cosine(q, embedding)
            if similarity < min_score:
                continue
            boost = _SEVERITY_BOOST.get(lesson["severity"], 0.0)
            boost += 0.05 * math.tanh(lesson["times_helpful"])
            lesson["relevance"] = round(similarity + boost, 4)
            scored.append(lesson)

        scored.sort(key=lambda item: item["relevance"], reverse=True)
        top = scored[:top_k]
        if reinforce and top:
            self.cog.lessons.reinforce_seen([item["id"] for item in top])
        return top

    def confirm_lesson(self, lesson_id: str) -> bool:
        """Record that a surfaced lesson actually helped, so it ranks higher."""
        if self.cog.lessons is None:
            return False
        return self.cog.lessons.confirm(lesson_id)

    def list_lessons(self, *, namespace: str, limit: int = 100) -> list[dict]:
        if self.cog.lessons is None:
            return []
        return self.cog.lessons.list(namespace=namespace, limit=limit)

    def preflight(self, *, task: str, namespace: str, limit: int = 5) -> dict:
        """Tell an agent what the collective knows *before* it starts a task.

        Bundles the relevant lessons (mistakes to avoid) with the best-matching
        procedures (playbooks that worked) into one pre-task briefing.
        """
        lessons = self.relevant_lessons(
            query=task, namespace=namespace, top_k=limit, reinforce=True
        )
        procedures = self.list_procedures(namespace=namespace, query=task, limit=limit)
        return {"task": task, "lessons": lessons, "procedures": procedures}

    # -- collaboration quality -------------------------------------------
    def collaboration_quality(self, *, namespace: str) -> dict:
        """A composite 0-100 score of how well agents are working together.

        It blends three observable signals into one trackable metric:
        recall hit-rate (memory is actually useful), feedback positivity
        (recalled memory leads to good outcomes), and conflict health (few
        unresolved contradictions). Watching it rise is how a team sees the
        shared memory paying off over time.
        """
        ep = self.episodes.stats(namespace)
        queries = ep["queries"]
        hit_rate = ep["query_hits"] / queries if queries else 0.0

        fb = self.cog.feedback.namespace_stats(namespace) if self.cog.feedback else {}
        fb_total = fb.get("total", 0)
        feedback_positivity = fb.get("positive", 0) / fb_total if fb_total else 0.0

        open_conflicts = self.cog.conflicts.count_open(namespace) if self.cog.conflicts else 0
        writes = ep["writes"]
        # one open conflict per write is the worst case; clamp to [0, 1].
        conflict_health = 1.0 - min(open_conflicts / writes, 1.0) if writes else 1.0

        agents = self.cog.agents.count(namespace) if self.cog.agents else 0
        score = 100.0 * (0.4 * hit_rate + 0.4 * feedback_positivity + 0.2 * conflict_health)
        return {
            "score": round(score, 1),
            "hit_rate": round(hit_rate, 4),
            "feedback_positivity": round(feedback_positivity, 4),
            "conflict_health": round(conflict_health, 4),
            "writes": writes,
            "queries": queries,
            "open_conflicts": open_conflicts,
            "agents": agents,
        }

    def list_conflicts(self, *, namespace: str | None = None, status: str | None = None) -> list:
        if self.cog.conflicts is None:
            return []
        return self.cog.conflicts.list(namespace=namespace, status=status)

    def resolve_conflict(self, conflict_id: str, *, winner_id: str | None) -> bool:
        """Resolve a contradiction and, when a winner is named, revise belief.

        Belief revision means the *losing* memory is trusted less: its confidence
        is decayed so recall demotes it, and repeated losses retire it entirely —
        the pool changes its mind in light of better information.
        """
        if self.cog.conflicts is None:
            return False
        row = self.cog.conflicts.get(conflict_id)
        if row is None:
            return False
        self.cog.conflicts.resolve(conflict_id, winner_id=winner_id)
        if winner_id and self.cog.belief_revision:
            loser_id = row["memory_id_b"] if winner_id == row["memory_id_a"] else row["memory_id_a"]
            self._decay_confidence(loser_id, namespace=row["namespace"])
        self.events.emit(
            "conflict.resolved",
            row["namespace"],
            {"conflict_id": conflict_id, "winner_id": winner_id},
        )
        return True

    def _decay_confidence(self, memory_id: str, *, namespace: str) -> None:
        """Lower a memory's trust after it loses a contradiction."""
        chunk = self.store.get(memory_id)
        if chunk is None:
            return
        decayed = chunk.provenance.confidence * self.cog.conflict_loser_decay
        self.store.set_confidence(memory_id, decayed)
        # Reinforce the demotion through the feedback channel as well, so the loss
        # also tells the re-ranker this memory led somewhere wrong.
        if self.cog.feedback is not None:
            self.cog.feedback.record(memory_id=memory_id, namespace=namespace, delta=-1.0)

    def distill_lessons(
        self,
        *,
        namespace: str,
        min_cluster: int,
        similarity: float,
        max_items: int = 200,
    ) -> dict:
        """Turn recurring failures into lesson drafts, with no human in the loop.

        Memories implicated in failed outcomes (net-negative feedback) are
        clustered; each recurring cluster becomes a lesson so the same mistake is
        flagged before it happens again.
        """
        if self.cog.lessons is None or self.cog.feedback is None:
            return {"scanned": 0, "clusters": 0, "lessons_created": 0}

        memory_ids = self.cog.feedback.negative(namespace=namespace, limit=max_items)
        records = [c for c in (self.store.get(mid) for mid in memory_ids) if c and c.dense]
        if len(records) < min_cluster:
            return {"scanned": len(records), "clusters": 0, "lessons_created": 0}

        vectors = [list(r.dense or []) for r in records]
        groups = cluster_by_similarity(vectors, threshold=similarity)

        created = 0
        used_clusters = 0
        for group in groups:
            if len(group) < min_cluster:
                continue
            used_clusters += 1
            merged = "\n".join(records[i].content for i in group)
            gist = self.summarizer.summarize(merged, max_sentences=2).strip()
            if not gist:
                continue
            # Skip if a near-identical lesson already exists.
            if self.relevant_lessons(
                query=gist, namespace=namespace, top_k=1, min_score=0.9, reinforce=False
            ):
                continue
            severity = "high" if len(group) >= min_cluster * 2 else "medium"
            self.record_lesson(
                namespace=namespace,
                trigger=gist,
                guidance=f"Recurring failure pattern — review before similar work: {gist}",
                severity=severity,
                created_by="auto-distiller",
            )
            created += 1
        return {
            "scanned": len(records),
            "clusters": used_clusters,
            "lessons_created": created,
        }

    def auto_resolve_conflicts(self, *, namespace: str, min_gap: float = 0.3) -> dict:
        """Close contradictions on their own when the evidence is decisive.

        For each open conflict, the two memories are weighed by an authority
        score (current trust plus accumulated feedback). When one clearly leads —
        or its rival has already been deleted — the conflict is resolved in its
        favour (which decays the loser). Ambiguous cases are left for a human.
        """
        if self.cog.conflicts is None:
            return {"resolved": 0, "skipped": 0}
        resolved = skipped = 0
        for conflict in self.cog.conflicts.list(namespace=namespace, status="open"):
            a = self.store.get(conflict["memory_id_a"])
            b = self.store.get(conflict["memory_id_b"])
            if a is None or b is None:
                # One side is already gone; the survivor wins by default.
                winner = conflict["memory_id_a"] if a else (conflict["memory_id_b"] if b else None)
                if winner is None:
                    skipped += 1
                    continue
                self.resolve_conflict(conflict["id"], winner_id=winner)
                resolved += 1
                continue
            scores = self.cog.feedback.scores([a.id, b.id]) if self.cog.feedback is not None else {}
            auth_a = a.provenance.confidence + 0.1 * math.tanh(scores.get(a.id, 0.0))
            auth_b = b.provenance.confidence + 0.1 * math.tanh(scores.get(b.id, 0.0))
            if abs(auth_a - auth_b) < min_gap:
                skipped += 1
                continue
            winner = a.id if auth_a > auth_b else b.id
            self.resolve_conflict(conflict["id"], winner_id=winner)
            resolved += 1
        return {"resolved": resolved, "skipped": skipped}

    def belief_timeline(self, *, query: str, namespace: str, limit: int = 50) -> list[dict]:
        """Trace how belief about a topic evolved — a memory diff over time.

        Returns the memories related to ``query`` in chronological order, each
        annotated with its current trust standing (active / demoted / retired),
        so one can see *which* claim fell out of favour and *when*.
        """
        dense = self.embedder.embed_query_dense(query)
        sparse = self.embedder.embed_query_sparse(query) if self.embedder.supports_sparse else None
        candidates = self.store.hybrid_search(
            dense=dense, sparse=sparse, limit=limit, namespace=namespace
        )
        events = [
            {
                "id": c.id,
                "date": (
                    datetime.fromtimestamp(c.provenance.created_at, tz=UTC).isoformat()
                    if c.provenance.created_at
                    else None
                ),
                "agent_id": c.provenance.agent_id,
                "content": c.content[:280],
                "confidence": round(c.provenance.confidence, 4),
                "status": _trust_status(c.provenance.confidence),
            }
            for c in candidates
        ]
        events.sort(key=lambda e: e["date"] or "")
        return events

    def namespace_health(self, *, namespace: str, scan_limit: int = 2000) -> dict:
        """A single pulse-check of a namespace's shared memory.

        Bundles volume, the trust distribution (active / demoted / retired),
        open contradictions, lesson count and the collaboration-quality score
        into one panel, so an operator can see the memory's health at a glance.
        """
        active = demoted = retired = total = 0
        confidence_sum = 0.0
        cursor: str | None = None
        remaining = scan_limit
        while remaining > 0:
            page = min(256, remaining)
            chunks, cursor = self.store.list_records(namespace=namespace, limit=page, cursor=cursor)
            for chunk in chunks:
                total += 1
                confidence_sum += chunk.provenance.confidence
                status = _trust_status(chunk.provenance.confidence)
                if status == "active":
                    active += 1
                elif status == "retired":
                    retired += 1
                else:
                    demoted += 1
            remaining -= page
            if not chunks or cursor is None:
                break

        ep = self.episodes.stats(namespace)
        return {
            "namespace": namespace,
            "memories": total,
            "trust": {"active": active, "demoted": demoted, "retired": retired},
            "avg_confidence": round(confidence_sum / total, 4) if total else 0.0,
            "writes": ep["writes"],
            "queries": ep["queries"],
            "open_conflicts": (
                self.cog.conflicts.count_open(namespace) if self.cog.conflicts else 0
            ),
            "lessons": self.cog.lessons.count(namespace) if self.cog.lessons else 0,
            "agents": self.cog.agents.count(namespace) if self.cog.agents else 0,
            "quality_score": self.collaboration_quality(namespace=namespace)["score"],
        }

    # -- scheduled maintenance -------------------------------------------
    def run_maintenance(
        self,
        *,
        auto_resolve: bool = True,
        consolidate: bool = False,
        distill_lessons: bool = False,
        min_gap: float = 0.3,
        consolidation_min_cluster: int = 3,
        consolidation_similarity: float = 0.83,
        lesson_distill_min_cluster: int = 2,
        lesson_distill_similarity: float = 0.83,
    ) -> dict:
        """Run one housekeeping cycle: sweep, then per-namespace upkeep.

        This is the shared brain's background "sleep": TTL-expired memories are
        purged globally, then each active namespace gets decisive contradictions
        auto-closed and (optionally) its memories consolidated and failure
        lessons distilled. Safe to call on a timer or on demand.
        """
        swept = self.sweep_expired()
        namespaces = self.episodes.namespaces()
        conflicts_resolved = insights = lessons_created = 0
        for namespace in namespaces:
            if auto_resolve:
                conflicts_resolved += self.auto_resolve_conflicts(
                    namespace=namespace, min_gap=min_gap
                )["resolved"]
            if consolidate:
                insights += self.consolidate(
                    namespace=namespace,
                    min_cluster=consolidation_min_cluster,
                    similarity=consolidation_similarity,
                )["insights"]
            if distill_lessons:
                lessons_created += self.distill_lessons(
                    namespace=namespace,
                    min_cluster=lesson_distill_min_cluster,
                    similarity=lesson_distill_similarity,
                )["lessons_created"]
        return {
            "swept": swept,
            "namespaces": len(namespaces),
            "conflicts_resolved": conflicts_resolved,
            "insights": insights,
            "lessons_created": lessons_created,
        }

    # -- portability (backup / migrate / share) --------------------------
    def export_namespace(self, *, namespace: str, scan_limit: int = 5000) -> dict:
        """Serialize a namespace's knowledge to a portable, vector-free document."""
        memories: list[dict] = []
        cursor: str | None = None
        remaining = scan_limit
        while remaining > 0:
            page = min(256, remaining)
            chunks, cursor = self.store.list_records(namespace=namespace, limit=page, cursor=cursor)
            for c in chunks:
                memories.append(
                    {
                        "content": c.content,
                        "agent_id": c.provenance.agent_id,
                        "session_id": c.provenance.session_id,
                        "task_id": c.provenance.task_id,
                        "source": c.provenance.source,
                        "confidence": c.provenance.confidence,
                        "tags": list(c.tags),
                        "metadata": dict(c.metadata),
                    }
                )
            remaining -= page
            if not chunks or cursor is None:
                break
        return {
            "namespace": namespace,
            "version": 1,
            "memories": memories,
            "lessons": self.list_lessons(namespace=namespace),
            "procedures": self.list_procedures(namespace=namespace),
        }

    def import_namespace(self, *, namespace: str, payload: dict) -> dict:
        """Recreate memories, lessons and procedures from an exported document.

        Vectors are re-embedded on the way in, so an export stays small and
        portable across embedding models.
        """
        memories = 0
        for m in payload.get("memories", []):
            result = self.write(
                content=m["content"],
                agent_id=m.get("agent_id", "import"),
                session_id=m.get("session_id", "import"),
                task_id=m.get("task_id"),
                namespace=namespace,
                tags=list(m.get("tags", [])),
                confidence=m.get("confidence", 1.0),
                metadata=dict(m.get("metadata", {})),
                source=m.get("source"),
                dedup=False,
            )
            memories += result.stored
        lessons = 0
        for lesson in payload.get("lessons", []):
            if self.record_lesson(
                namespace=namespace,
                trigger=lesson["trigger"],
                guidance=lesson["guidance"],
                severity=lesson.get("severity", "medium"),
                created_by=lesson.get("created_by"),
            ):
                lessons += 1
        procedures = 0
        for proc in payload.get("procedures", []):
            if self.create_procedure(
                namespace=namespace,
                title=proc["title"],
                steps=list(proc.get("steps", [])),
                tags=list(proc.get("tags", [])),
                created_by=proc.get("created_by"),
            ):
                procedures += 1
        return {"memories": memories, "lessons": lessons, "procedures": procedures}

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
        include_dates: bool = False,
        since: float | None = None,
        until: float | None = None,
        with_lessons: bool = True,
    ) -> AssembledContext:
        params = RetrievalParams(
            top_k=top_k or self.defaults.top_k,
            token_budget=token_budget or self.defaults.token_budget,
            candidate_pool=self.defaults.candidate_pool,
            mmr_lambda=self.defaults.mmr_lambda,
            rerank=rerank,
            expand_parents=expand_parents,
            include_dates=include_dates,
            since=since,
            until=until,
        )
        with span("memory.query", namespace=namespace, top_k=params.top_k):
            result = self.retriever.retrieve(
                query, namespace=namespace, filters=filters, params=params
            )
        if with_lessons and self.cog.lessons_enabled:
            self._attach_lessons(result, query=query, namespace=namespace)
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

    def _attach_lessons(self, result: AssembledContext, *, query: str, namespace: str) -> None:
        """Raise relevant past-mistake guardrails on top of recalled context."""
        lessons = self.relevant_lessons(query=query, namespace=namespace)
        if not lessons:
            return
        result.lessons = lessons
        banner_lines = ["[!] Lessons from past mistakes (heed before acting):"]
        banner_lines += [f"- ({lesson['severity']}) {lesson['guidance']}" for lesson in lessons]
        banner = "\n".join(banner_lines)
        result.context = f"{banner}\n\n{result.context}" if result.context else banner
        result.tokens_used += count_tokens(banner)

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
        if namespace and self.cog.lessons is not None:
            self.cog.lessons.delete_by_namespace(namespace)
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
