"""Data-access helpers for episodic memory."""

from __future__ import annotations

from sqlalchemy import select

from context_bridge.core.models import new_id
from context_bridge.db.models import Episode
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
