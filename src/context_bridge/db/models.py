"""SQLAlchemy models for episodic / provenance memory.

The vector store answers "what is similar?"; this relational layer answers
"who did what, when, and why?". Every write and summarisation is logged as an
:class:`Episode`, giving a queryable, auditable task timeline that pure vector
search cannot reconstruct.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Episode(Base):
    """A single recorded event in a session's life."""

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # write | query | summary
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chunk_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        Index("ix_episodes_session_created", "session_id", "created_at"),
        Index("ix_episodes_namespace", "namespace"),
    )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "namespace": self.namespace,
            "kind": self.kind,
            "content": self.content,
            "chunk_ids": list(self.chunk_ids or []),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ParentDocument(Base):
    """The full source text for a set of child chunks (small-to-big).

    Storing the parent once here — rather than duplicating it inside every
    child chunk's vector payload — keeps the vector store lean while still
    allowing retrieval to expand a matched chunk back to its broader context.
    """

    __tablename__ = "parent_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
