"""SQLAlchemy models for episodic / provenance memory.

The vector store answers "what is similar?"; this relational layer answers
"who did what, when, and why?". Every write and summarisation is logged as an
:class:`Episode`, giving a queryable, auditable task timeline that pure vector
search cannot reconstruct.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
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


class Feedback(Base):
    """Aggregated outcome feedback per memory, used to re-rank recall."""

    __tablename__ = "feedback"

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    votes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class Conflict(Base):
    """A detected contradiction between two memories (truth-maintenance)."""

    __tablename__ = "conflicts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    memory_id_a: Mapped[str] = mapped_column(String(64), nullable=False)
    memory_id_b: Mapped[str] = mapped_column(String(64), nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")  # open|resolved
    winner_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_conflicts_namespace_status", "namespace", "status"),)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "memory_id_a": self.memory_id_a,
            "memory_id_b": self.memory_id_b,
            "similarity": self.similarity,
            "status": self.status,
            "winner_id": self.winner_id,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }


class GraphNode(Base):
    """An entity extracted from memory content (knowledge-graph node)."""

    __tablename__ = "graph_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="entity")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (UniqueConstraint("namespace", "name", name="uq_graph_node"),)


class GraphEdge(Base):
    """A relation between two entities, attributed to a source memory."""

    __tablename__ = "graph_edges"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    relation: Mapped[str] = mapped_column(String(128), nullable=False)
    target: Mapped[str] = mapped_column(String(256), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_graph_edges_ns_source", "namespace", "source"),)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "memory_id": self.memory_id,
            "namespace": self.namespace,
        }


class AgentProfile(Base):
    """Per-namespace reputation for an agent — who is good at what."""

    __tablename__ = "agent_profiles"

    id: Mapped[str] = mapped_column(String(320), primary_key=True)  # namespace::agent_id
    namespace: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    writes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    useful: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unhelpful: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_agent_profiles_ns_score", "namespace", "score"),)

    @property
    def reputation(self) -> float:
        total = self.useful + self.unhelpful
        return self.useful / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "writes": self.writes,
            "useful": self.useful,
            "unhelpful": self.unhelpful,
            "score": self.score,
            "reputation": round(self.reputation, 4),
        }


class Procedure(Base):
    """A reusable playbook distilled from successful work (procedural memory)."""

    __tablename__ = "procedures"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(256), nullable=False, default="default")
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_procedures_namespace", "namespace"),)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "title": self.title,
            "steps": list(self.steps or []),
            "tags": list(self.tags or []),
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "success_rate": round(self.success_rate, 4),
            "created_by": self.created_by,
        }
