"""Pydantic request/response models for the agent-facing API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from context_bridge.core.models import AssembledContext, RetrievedChunk


class WriteRequest(BaseModel):
    content: str = Field(..., min_length=1)
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


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    namespace: str = "default"
    agent_id: str = "system"
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    token_budget: int | None = Field(default=None, ge=1)
    filters: dict | None = None
    rerank: bool = True
    expand_parents: bool = False


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
        )


class QueryResponse(BaseModel):
    context: str
    tokens_used: int
    chunks: list[ChunkOut]
    sources: list[dict]

    @classmethod
    def from_assembled(cls, assembled: AssembledContext) -> QueryResponse:
        return cls(
            context=assembled.context,
            tokens_used=assembled.tokens_used,
            chunks=[ChunkOut.from_chunk(c) for c in assembled.chunks],
            sources=assembled.sources,
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


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
