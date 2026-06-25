"""Pydantic request/response models for the agent-facing API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from context_bridge.core.models import AssembledContext, RetrievedChunk

# Hard caps to bound memory use / abuse. A single memory should be one document;
# split larger inputs client-side.
MAX_CONTENT_CHARS = 1_000_000
MAX_QUERY_CHARS = 16_000


class WriteRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)
    agent_id: str
    session_id: str
    task_id: str | None = None
    namespace: str = "default"
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)
    source: str | None = None
    ttl_seconds: int | None = None
    dedup: bool = True
    summarize_before_store: bool = False


class WriteResponse(BaseModel):
    ids: list[str]
    stored: int
    deduped: int
    skipped: bool = False


class WriteBatchRequest(BaseModel):
    items: list[WriteRequest] = Field(..., min_length=1, max_length=256)


class WriteBatchResponse(BaseModel):
    results: list[WriteResponse]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    namespace: str = "default"
    agent_id: str = "system"
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    token_budget: int | None = Field(default=None, ge=1)
    filters: dict | None = None
    rerank: bool = True
    expand_parents: bool = False
    include_dates: bool = False
    since: float | None = None  # epoch seconds; only recall memories at/after this
    until: float | None = None  # epoch seconds; only recall memories at/before this
    with_lessons: bool = True  # raise relevant past-mistake guardrails on recall


def _reason_from_signals(signals: dict) -> str:
    """Render a short, human-readable 'why this was recalled' explanation."""
    parts: list[str] = []
    match = signals.get("match")
    if match is not None:
        parts.append(f"relevance {match}")
    feedback = signals.get("feedback")
    if feedback:
        parts.append("boosted by past success" if feedback > 0 else "penalised by past failure")
    confidence = signals.get("confidence")
    if confidence is not None and confidence < 1.0:
        parts.append("demoted (low confidence)")
    age = signals.get("age_days")
    if age is not None:
        parts.append("recent" if age <= 1 else f"{age:g}d old")
    return "; ".join(parts)


class ChunkOut(BaseModel):
    id: str
    content: str
    score: float
    namespace: str
    agent_id: str
    session_id: str
    task_id: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    signals: dict = Field(default_factory=dict)
    reason: str = ""

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> ChunkOut:
        return cls(
            id=chunk.id,
            content=chunk.content,
            score=chunk.score,
            namespace=chunk.namespace,
            agent_id=chunk.provenance.agent_id,
            session_id=chunk.provenance.session_id,
            task_id=chunk.provenance.task_id,
            source=chunk.provenance.source,
            tags=chunk.tags,
            signals=chunk.signals,
            reason=_reason_from_signals(chunk.signals),
        )


class QueryResponse(BaseModel):
    context: str
    tokens_used: int
    chunks: list[ChunkOut]
    sources: list[dict]
    lessons: list[dict] = Field(default_factory=list)

    @classmethod
    def from_assembled(cls, assembled: AssembledContext) -> QueryResponse:
        return cls(
            context=assembled.context,
            tokens_used=assembled.tokens_used,
            chunks=[ChunkOut.from_chunk(c) for c in assembled.chunks],
            sources=assembled.sources,
            lessons=assembled.lessons,
        )


class SummarizeRequest(BaseModel):
    session_id: str
    namespace: str = "default"
    agent_id: str = "summarizer"
    max_sentences: int = Field(default=5, ge=1, le=50)
    store_summary: bool = True


class SummarizeResponse(BaseModel):
    summary: str
    chunk_ids: list[str]


class TimelineResponse(BaseModel):
    session_id: str
    episodes: list[dict]


class TurnRequest(BaseModel):
    agent_id: str
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)
    kind: str = "note"


class TurnsResponse(BaseModel):
    session_id: str
    turns: list[dict]


class DistillRequest(BaseModel):
    namespace: str = "default"
    agent_id: str = "distiller"
    max_promote: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=1.0, ge=0)


class DistillResponse(BaseModel):
    scanned: int
    promoted: int
    ids: list[str]


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


class ListResponse(BaseModel):
    chunks: list[ChunkOut]
    next_cursor: str | None = None


class SweepResponse(BaseModel):
    deleted: int


class ForgetResponse(BaseModel):
    vectors_deleted: int
    episodes_deleted: int
    parents_deleted: int


class FeedbackRequest(BaseModel):
    memory_id: str
    namespace: str = "default"
    useful: bool
    weight: float = Field(default=1.0, gt=0, le=10)


class ConsolidateResponse(BaseModel):
    scanned: int
    clusters: int
    insights: int


class ConflictResolveRequest(BaseModel):
    winner_id: str | None = None


class GraphEdgeOut(BaseModel):
    source: str
    relation: str
    target: str
    memory_id: str | None = None


class GraphResponse(BaseModel):
    entity: str
    edges: list[GraphEdgeOut]


class AliasRequest(BaseModel):
    namespace: str = "default"
    alias: str = Field(..., min_length=1, max_length=256)
    canonical: str = Field(..., min_length=1, max_length=256)


class AliasResponse(BaseModel):
    registered: bool
    aliases: list[dict]


class AlignRequest(BaseModel):
    namespace: str = "default"


class AlignResponse(BaseModel):
    groups_merged: int
    aliases_created: int


class QualityResponse(BaseModel):
    score: float
    hit_rate: float
    feedback_positivity: float
    conflict_health: float
    writes: int
    queries: int
    open_conflicts: int
    agents: int


class AutoResolveResponse(BaseModel):
    resolved: int
    skipped: int


class HealthPanelResponse(BaseModel):
    namespace: str
    memories: int
    trust: dict
    avg_confidence: float
    writes: int
    queries: int
    open_conflicts: int
    lessons: int
    agents: int
    quality_score: float


class BeliefTimelineResponse(BaseModel):
    query: str
    events: list[dict]


class OutcomeRequest(BaseModel):
    session_id: str
    namespace: str = "default"
    success: bool
    weight: float = Field(default=1.0, gt=0, le=10)
    # Optionally capture a lesson learned from this outcome (typically a failure).
    lesson: str | None = Field(default=None, max_length=MAX_CONTENT_CHARS)
    lesson_trigger: str | None = Field(default=None, max_length=MAX_QUERY_CHARS)
    severity: Literal["low", "medium", "high"] = "medium"


class OutcomeResponse(BaseModel):
    memories_credited: int
    agents_credited: int
    success: bool
    lesson_id: str | None = None


class LessonRequest(BaseModel):
    namespace: str = "default"
    trigger: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    guidance: str = Field(..., min_length=1, max_length=MAX_CONTENT_CHARS)
    severity: Literal["low", "medium", "high"] = "medium"
    created_by: str | None = None
    session_id: str | None = None


class LessonCreated(BaseModel):
    id: str


class LessonsResponse(BaseModel):
    lessons: list[dict]


class LessonsDistilled(BaseModel):
    scanned: int
    clusters: int
    lessons_created: int


class PreflightRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    namespace: str = "default"
    limit: int = Field(default=5, ge=1, le=50)


class PreflightResponse(BaseModel):
    task: str
    lessons: list[dict]
    procedures: list[dict]


class AgentsResponse(BaseModel):
    agents: list[dict]


class ProcedureCreate(BaseModel):
    namespace: str = "default"
    title: str = Field(..., min_length=1, max_length=512)
    steps: list[str] = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    created_by: str | None = None


class ProcedureCreated(BaseModel):
    id: str


class ProceduresResponse(BaseModel):
    procedures: list[dict]


class ProcedureOutcomeRequest(BaseModel):
    success: bool
