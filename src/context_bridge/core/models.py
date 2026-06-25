"""Domain models shared across the memory pipeline.

These are deliberately framework-agnostic dataclasses so that the chunking,
embedding, vector-store and retrieval layers can pass typed objects around
without importing pydantic or SQLAlchemy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def new_id() -> str:
    """Generate a fresh opaque identifier.

    Returns a canonical UUID string (with dashes) so the value is accepted
    directly as a Qdrant point id.
    """
    return str(uuid.uuid4())


def now_ts() -> float:
    """Current wall-clock time as a UNIX timestamp."""
    return time.time()


@dataclass(slots=True)
class SparseVector:
    """A sparse embedding expressed as parallel index/value arrays."""

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.indices


@dataclass(slots=True)
class Provenance:
    """Where a memory came from and how much we trust it.

    Provenance is what lets the shared pool stay healthy: every stored chunk
    is attributable to an agent / task / session, carries a confidence score
    and an optional time-to-live so stale or low-quality memories can decay.
    """

    agent_id: str
    session_id: str
    task_id: str | None = None
    source: str | None = None
    confidence: float = 1.0
    created_at: float = field(default_factory=now_ts)
    ttl_seconds: int | None = None

    def is_expired(self, *, at: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return False
        return (at or now_ts()) - self.created_at > self.ttl_seconds


@dataclass(slots=True)
class Chunk:
    """A unit of text produced by a chunker, ready to be embedded.

    ``parent_id`` enables the small-to-big strategy: a small, precise chunk is
    indexed for retrieval while ``parent_text`` carries the broader context to
    expand into once the chunk is selected.
    """

    text: str
    index: int
    parent_id: str
    parent_text: str | None = None
    id: str = field(default_factory=new_id)


@dataclass(slots=True)
class MemoryRecord:
    """A fully prepared record about to be persisted to the vector store."""

    id: str
    content: str
    namespace: str
    provenance: Provenance
    parent_id: str
    parent_text: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    dense: list[float] | None = None
    sparse: SparseVector | None = None


@dataclass(slots=True)
class RetrievedChunk:
    """A candidate returned by the vector store / retrieval pipeline."""

    id: str
    content: str
    score: float
    namespace: str
    provenance: Provenance
    parent_id: str
    parent_text: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    dense: list[float] | None = None


@dataclass(slots=True)
class AssembledContext:
    """The final, budget-bounded answer returned to a calling agent."""

    context: str
    chunks: list[RetrievedChunk]
    tokens_used: int
    # Guardrails raised from past mistakes relevant to this query (see Lesson).
    lessons: list[dict] = field(default_factory=list)

    @property
    def sources(self) -> list[dict]:
        """Compact citation list derived from the included chunks."""
        out: list[dict] = []
        for c in self.chunks:
            out.append(
                {
                    "id": c.id,
                    "agent_id": c.provenance.agent_id,
                    "session_id": c.provenance.session_id,
                    "task_id": c.provenance.task_id,
                    "source": c.provenance.source,
                    "score": c.score,
                }
            )
        return out
