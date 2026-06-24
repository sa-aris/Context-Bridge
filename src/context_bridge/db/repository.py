"""Data-access helpers for episodic memory."""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import delete, func, select

from context_bridge.core.graph.resolver import choose_canonical, normalize
from context_bridge.core.models import new_id
from context_bridge.db.models import (
    AgentProfile,
    Conflict,
    EntityAlias,
    Episode,
    Feedback,
    GraphEdge,
    GraphNode,
    ParentDocument,
    Procedure,
)
from context_bridge.db.session import Database


class EpisodeRepository:
    """CRUD-ish access to the :class:`Episode` table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        *,
        session_id: str,
        agent_id: str,
        kind: str,
        content: str = "",
        task_id: str | None = None,
        namespace: str = "default",
        chunk_ids: list[str] | None = None,
        confidence: float = 1.0,
    ) -> str:
        """Persist an episode and return its id."""
        episode_id = new_id()
        with self.db.session() as session:
            session.add(
                Episode(
                    id=episode_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    namespace=namespace,
                    kind=kind,
                    content=content,
                    chunk_ids=list(chunk_ids or []),
                    confidence=confidence,
                )
            )
        return episode_id

    def timeline(self, session_id: str, *, limit: int = 100) -> list[dict]:
        """Return episodes for a session, oldest first."""
        stmt = (
            select(Episode)
            .where(Episode.session_id == session_id)
            .order_by(Episode.created_at.asc(), Episode.id.asc())
            .limit(limit)
        )
        with self.db.session() as session:
            return [e.as_dict() for e in session.scalars(stmt)]

    def delete_by(self, *, namespace: str | None = None, session_id: str | None = None) -> int:
        """Delete episodes matching a namespace and/or session; return the count."""
        stmt = delete(Episode)
        if namespace:
            stmt = stmt.where(Episode.namespace == namespace)
        if session_id:
            stmt = stmt.where(Episode.session_id == session_id)
        with self.db.session() as session:
            return session.execute(stmt).rowcount or 0  # type: ignore[attr-defined]

    def stats(self, namespace: str) -> dict:
        """Counts used by the collaboration-quality score."""
        with self.db.session() as session:
            writes = session.scalar(
                select(func.count())
                .select_from(Episode)
                .where(Episode.namespace == namespace, Episode.kind == "write")
            )
            queries = session.scalar(
                select(func.count())
                .select_from(Episode)
                .where(Episode.namespace == namespace, Episode.kind == "query")
            )
            hits = 0
            q_stmt = select(Episode).where(Episode.namespace == namespace, Episode.kind == "query")
            for ep in session.scalars(q_stmt):
                if ep.chunk_ids:
                    hits += 1
        return {"writes": int(writes or 0), "queries": int(queries or 0), "query_hits": hits}


class ParentRepository:
    """Stores and retrieves parent documents for the small-to-big strategy."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, *, parent_id: str, namespace: str, text: str) -> None:
        with self.db.session() as session:
            existing = session.get(ParentDocument, parent_id)
            if existing is not None:
                existing.text = text
                existing.namespace = namespace
            else:
                session.add(ParentDocument(id=parent_id, namespace=namespace, text=text))

    def get_texts(self, parent_ids: list[str]) -> dict[str, str]:
        if not parent_ids:
            return {}
        stmt = select(ParentDocument).where(ParentDocument.id.in_(set(parent_ids)))
        with self.db.session() as session:
            return {p.id: p.text for p in session.scalars(stmt)}

    def delete_by_namespace(self, namespace: str) -> int:
        with self.db.session() as session:
            return (
                session.execute(  # type: ignore[attr-defined]
                    delete(ParentDocument).where(ParentDocument.namespace == namespace)
                ).rowcount
                or 0
            )


class FeedbackRepository:
    """Aggregated outcome feedback used to re-rank recall over time."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, *, memory_id: str, namespace: str, delta: float) -> None:
        with self.db.session() as session:
            row = session.get(Feedback, memory_id)
            if row is None:
                session.add(
                    Feedback(memory_id=memory_id, namespace=namespace, score=delta, votes=1)
                )
            else:
                row.score += delta
                row.votes += 1

    def scores(self, memory_ids: list[str]) -> dict[str, float]:
        if not memory_ids:
            return {}
        stmt = select(Feedback).where(Feedback.memory_id.in_(set(memory_ids)))
        with self.db.session() as session:
            return {f.memory_id: f.score for f in session.scalars(stmt)}

    def namespace_stats(self, namespace: str) -> dict:
        stmt = select(Feedback).where(Feedback.namespace == namespace)
        positive = total = 0
        with self.db.session() as session:
            for row in session.scalars(stmt):
                total += 1
                if row.score > 0:
                    positive += 1
        return {"positive": positive, "total": total}


class ConflictRepository:
    """Records and resolves detected contradictions (truth-maintenance)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, *, namespace: str, memory_id_a: str, memory_id_b: str, similarity: float) -> str:
        conflict_id = new_id()
        with self.db.session() as session:
            session.add(
                Conflict(
                    id=conflict_id,
                    namespace=namespace,
                    memory_id_a=memory_id_a,
                    memory_id_b=memory_id_b,
                    similarity=similarity,
                )
            )
        return conflict_id

    def list(self, *, namespace: str | None = None, status: str | None = None) -> list[dict]:
        stmt = select(Conflict)
        if namespace:
            stmt = stmt.where(Conflict.namespace == namespace)
        if status:
            stmt = stmt.where(Conflict.status == status)
        stmt = stmt.order_by(Conflict.detected_at.desc())
        with self.db.session() as session:
            return [c.as_dict() for c in session.scalars(stmt)]

    def resolve(self, conflict_id: str, *, winner_id: str | None) -> bool:
        with self.db.session() as session:
            row = session.get(Conflict, conflict_id)
            if row is None:
                return False
            row.status = "resolved"
            row.winner_id = winner_id
            return True

    def count_open(self, namespace: str) -> int:
        with self.db.session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(Conflict)
                    .where(Conflict.namespace == namespace, Conflict.status == "open")
                )
                or 0
            )


class GraphRepository:
    """A lightweight knowledge graph (entities + relations) over memory."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _alias_key(namespace: str, alias: str) -> str:
        return f"{namespace}::{normalize(alias)}"

    def _alias_map(self, session, namespace: str) -> dict[str, str]:
        """Return ``{normalized_alias: canonical}`` for a namespace."""
        stmt = select(EntityAlias).where(EntityAlias.namespace == namespace)
        return {row.alias: row.canonical for row in session.scalars(stmt)}

    @staticmethod
    def _resolve(name: str, alias_map: dict[str, str]) -> str:
        return alias_map.get(normalize(name), name)

    def _ensure_node(self, session, namespace: str, name: str) -> None:
        exists = session.scalar(
            select(GraphNode).where(GraphNode.namespace == namespace, GraphNode.name == name)
        )
        if exists is None:
            session.add(GraphNode(id=new_id(), namespace=namespace, name=name))

    def add_edge(
        self, *, namespace: str, source: str, relation: str, target: str, memory_id: str | None
    ) -> None:
        with self.db.session() as session:
            alias_map = self._alias_map(session, namespace)
            source = self._resolve(source, alias_map)
            target = self._resolve(target, alias_map)
            self._ensure_node(session, namespace, source)
            self._ensure_node(session, namespace, target)
            session.add(
                GraphEdge(
                    id=new_id(),
                    namespace=namespace,
                    source=source,
                    relation=relation,
                    target=target,
                    memory_id=memory_id,
                )
            )

    def register_alias(self, *, namespace: str, alias: str, canonical: str) -> bool:
        """Map ``alias`` to ``canonical`` and rewrite existing nodes/edges.

        Returns ``False`` when the alias is empty or already resolves to the
        same canonical name (a no-op).
        """
        if not normalize(alias) or normalize(alias) == normalize(canonical):
            return False
        key = self._alias_key(namespace, alias)
        with self.db.session() as session:
            row = session.get(EntityAlias, key)
            if row is None:
                session.add(
                    EntityAlias(
                        id=key, namespace=namespace, alias=normalize(alias), canonical=canonical
                    )
                )
            else:
                row.canonical = canonical
            self._merge_into(session, namespace, alias=alias, canonical=canonical)
        return True

    def _merge_into(self, session, namespace: str, *, alias: str, canonical: str) -> None:
        """Rewrite edges off ``alias`` onto ``canonical`` and drop the stale node."""
        self._ensure_node(session, namespace, canonical)
        for edge in session.scalars(select(GraphEdge).where(GraphEdge.namespace == namespace)):
            if normalize(edge.source) == normalize(alias):
                edge.source = canonical
            if normalize(edge.target) == normalize(alias):
                edge.target = canonical
        session.execute(
            delete(GraphNode).where(GraphNode.namespace == namespace, GraphNode.name == alias)
        )

    def align(self, namespace: str) -> dict:
        """Merge surface variants of the same entity onto one canonical name.

        Nodes are grouped by their normalized name; within each group the most
        connected variant wins (ties broken by brevity, then alphabetically).
        Losing variants become aliases and their edges are rewritten.
        """
        groups_merged = aliases_created = 0
        with self.db.session() as session:
            nodes = list(session.scalars(select(GraphNode).where(GraphNode.namespace == namespace)))
            groups: dict[str, list[str]] = defaultdict(list)
            for node in nodes:
                groups[normalize(node.name)].append(node.name)

            edge_counts: Counter[str] = Counter()
            for edge in session.scalars(select(GraphEdge).where(GraphEdge.namespace == namespace)):
                edge_counts[edge.source] += 1
                edge_counts[edge.target] += 1

            for variants in groups.values():
                if len(variants) < 2:
                    continue
                groups_merged += 1
                canonical = choose_canonical(variants, dict(edge_counts))
                for variant in variants:
                    if variant == canonical:
                        continue
                    key = self._alias_key(namespace, variant)
                    if session.get(EntityAlias, key) is None:
                        session.add(
                            EntityAlias(
                                id=key,
                                namespace=namespace,
                                alias=normalize(variant),
                                canonical=canonical,
                            )
                        )
                        aliases_created += 1
                    self._merge_into(session, namespace, alias=variant, canonical=canonical)
        return {"groups_merged": groups_merged, "aliases_created": aliases_created}

    def list_aliases(self, namespace: str) -> list[dict]:
        stmt = (
            select(EntityAlias)
            .where(EntityAlias.namespace == namespace)
            .order_by(EntityAlias.canonical.asc(), EntityAlias.alias.asc())
        )
        with self.db.session() as session:
            return [
                {"alias": row.alias, "canonical": row.canonical} for row in session.scalars(stmt)
            ]

    def neighbors(self, *, namespace: str, entity: str, hops: int = 1) -> list[dict]:
        """Breadth-first traversal from ``entity`` up to ``hops`` edges out."""
        seen: set[str] = {entity}
        frontier = {entity}
        edges: list[dict] = []
        edge_ids: set[tuple] = set()
        with self.db.session() as session:
            for _ in range(max(hops, 1)):
                if not frontier:
                    break
                stmt = select(GraphEdge).where(
                    GraphEdge.namespace == namespace,
                    GraphEdge.source.in_(frontier) | GraphEdge.target.in_(frontier),
                )
                next_frontier: set[str] = set()
                for edge in session.scalars(stmt):
                    key = (edge.source, edge.relation, edge.target)
                    if key not in edge_ids:
                        edge_ids.add(key)
                        edges.append(edge.as_dict())
                    for node in (edge.source, edge.target):
                        if node not in seen:
                            seen.add(node)
                            next_frontier.add(node)
                frontier = next_frontier
        return edges

    def delete_by_namespace(self, namespace: str) -> int:
        with self.db.session() as session:
            edges = (
                session.execute(  # type: ignore[attr-defined]
                    delete(GraphEdge).where(GraphEdge.namespace == namespace)
                ).rowcount
                or 0
            )
            session.execute(delete(GraphNode).where(GraphNode.namespace == namespace))
            session.execute(delete(EntityAlias).where(EntityAlias.namespace == namespace))
        return edges


class AgentProfileRepository:
    """Per-namespace agent reputation (writes + outcome-weighted score)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _key(namespace: str, agent_id: str) -> str:
        return f"{namespace}::{agent_id}"

    def _get_or_create(self, session, namespace: str, agent_id: str) -> AgentProfile:
        key = self._key(namespace, agent_id)
        row = session.get(AgentProfile, key)
        if row is None:
            row = AgentProfile(
                id=key,
                namespace=namespace,
                agent_id=agent_id,
                writes=0,
                useful=0,
                unhelpful=0,
                score=0.0,
            )
            session.add(row)
        return row

    def record_write(self, *, namespace: str, agent_id: str) -> None:
        with self.db.session() as session:
            self._get_or_create(session, namespace, agent_id).writes += 1

    def record_outcome(self, *, namespace: str, agent_id: str, delta: float, useful: bool) -> None:
        with self.db.session() as session:
            row = self._get_or_create(session, namespace, agent_id)
            row.score += delta
            if useful:
                row.useful += 1
            else:
                row.unhelpful += 1

    def top(self, *, namespace: str, limit: int = 20) -> list[dict]:
        stmt = (
            select(AgentProfile)
            .where(AgentProfile.namespace == namespace)
            .order_by(AgentProfile.score.desc())
            .limit(limit)
        )
        with self.db.session() as session:
            return [p.as_dict() for p in session.scalars(stmt)]

    def count(self, namespace: str) -> int:
        with self.db.session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(AgentProfile)
                    .where(AgentProfile.namespace == namespace)
                )
                or 0
            )

    def delete_by_namespace(self, namespace: str) -> int:
        with self.db.session() as session:
            return (
                session.execute(  # type: ignore[attr-defined]
                    delete(AgentProfile).where(AgentProfile.namespace == namespace)
                ).rowcount
                or 0
            )


class ProcedureRepository:
    """Reusable playbooks with success/failure tracking (procedural memory)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        *,
        namespace: str,
        title: str,
        steps: list[str],
        tags: list[str] | None = None,
        created_by: str | None = None,
    ) -> str:
        proc_id = new_id()
        with self.db.session() as session:
            session.add(
                Procedure(
                    id=proc_id,
                    namespace=namespace,
                    title=title,
                    steps=list(steps),
                    tags=list(tags or []),
                    created_by=created_by,
                )
            )
        return proc_id

    def list(self, *, namespace: str, query: str | None = None, limit: int = 50) -> list[dict]:
        stmt = select(Procedure).where(Procedure.namespace == namespace)
        if query:
            stmt = stmt.where(Procedure.title.ilike(f"%{query}%"))
        stmt = stmt.order_by(Procedure.success_count.desc(), Procedure.created_at.desc()).limit(
            limit
        )
        with self.db.session() as session:
            return [p.as_dict() for p in session.scalars(stmt)]

    def record_outcome(self, procedure_id: str, *, success: bool) -> bool:
        with self.db.session() as session:
            row = session.get(Procedure, procedure_id)
            if row is None:
                return False
            if success:
                row.success_count += 1
            else:
                row.fail_count += 1
            return True

    def delete_by_namespace(self, namespace: str) -> int:
        with self.db.session() as session:
            return (
                session.execute(  # type: ignore[attr-defined]
                    delete(Procedure).where(Procedure.namespace == namespace)
                ).rowcount
                or 0
            )
