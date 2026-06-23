"""Data-access helpers for episodic memory."""

from __future__ import annotations

from sqlalchemy import delete, select

from context_bridge.core.models import new_id
from context_bridge.db.models import Episode, ParentDocument
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
