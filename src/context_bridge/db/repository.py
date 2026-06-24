"""Data-access helpers for episodic memory."""

from __future__ import annotations

from sqlalchemy import delete, select

from context_bridge.core.models import new_id
from context_bridge.db.models import (
    AgentProfile,
    Conflict,
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


class GraphRepository:
    """A lightweight knowledge graph (entities + relations) over memory."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_edge(
        self, *, namespace: str, source: str, relation: str, target: str, memory_id: str | None
    ) -> None:
        with self.db.session() as session:
            for name in (source, target):
                exists = session.scalar(
                    select(GraphNode).where(
                        GraphNode.namespace == namespace, GraphNode.name == name
                    )
                )
                if exists is None:
                    session.add(GraphNode(id=new_id(), namespace=namespace, name=name))
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
