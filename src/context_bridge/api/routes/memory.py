"""Memory read/write endpoints — the core agent-facing contract."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from context_bridge.api import metrics
from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import (
    ChunkOut,
    ForgetResponse,
    ListResponse,
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
    WriteBatchRequest,
    WriteBatchResponse,
    WriteRequest,
    WriteResponse,
)
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/memory", tags=["memory"])


def _do_write(manager: MemoryManager, req: WriteRequest) -> WriteResponse:
    result = manager.write(
        content=req.content,
        agent_id=req.agent_id,
        session_id=req.session_id,
        task_id=req.task_id,
        namespace=req.namespace,
        tags=req.tags,
        confidence=req.confidence,
        metadata=req.metadata,
        source=req.source,
        ttl_seconds=req.ttl_seconds,
        dedup=req.dedup,
        summarize_before_store=req.summarize_before_store,
    )
    metrics.WRITES.inc()
    metrics.CHUNKS_STORED.inc(result.stored)
    metrics.CHUNKS_DEDUPED.inc(result.deduped)
    return WriteResponse(
        ids=result.ids, stored=result.stored, deduped=result.deduped, skipped=result.skipped
    )


@router.post("/write", response_model=WriteResponse)
def write_memory(
    req: WriteRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> WriteResponse:
    authorize(request, req.namespace, "write")
    return _do_write(manager, req)


@router.post("/write_batch", response_model=WriteBatchResponse)
def write_memory_batch(
    req: WriteBatchRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> WriteBatchResponse:
    for item in req.items:
        authorize(request, item.namespace, "write")
    return WriteBatchResponse(results=[_do_write(manager, item) for item in req.items])


@router.post("/query", response_model=QueryResponse)
def query_memory(
    req: QueryRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> QueryResponse:
    authorize(request, req.namespace, "read")
    assembled = manager.query(
        query=req.query,
        namespace=req.namespace,
        agent_id=req.agent_id,
        session_id=req.session_id,
        top_k=req.top_k,
        token_budget=req.token_budget,
        filters=req.filters,
        rerank=req.rerank,
        expand_parents=req.expand_parents,
    )
    metrics.QUERIES.inc()
    metrics.QUERY_TOKENS.observe(assembled.tokens_used)
    metrics.QUERY_CHUNKS.observe(len(assembled.chunks))
    return QueryResponse.from_assembled(assembled)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/query/stream")
def query_memory_stream(
    req: QueryRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> StreamingResponse:
    """Recall as a Server-Sent Events stream: one event per chunk, then 'done'.

    Lets an agent render or forward context progressively instead of waiting
    for the whole assembled block.
    """
    authorize(request, req.namespace, "read")
    assembled = manager.query(
        query=req.query,
        namespace=req.namespace,
        agent_id=req.agent_id,
        session_id=req.session_id,
        top_k=req.top_k,
        token_budget=req.token_budget,
        filters=req.filters,
        rerank=req.rerank,
        expand_parents=req.expand_parents,
    )
    metrics.QUERIES.inc()
    metrics.QUERY_TOKENS.observe(assembled.tokens_used)
    metrics.QUERY_CHUNKS.observe(len(assembled.chunks))

    def generate() -> Iterator[str]:
        for chunk in assembled.chunks:
            yield _sse(
                "chunk",
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "agent_id": chunk.provenance.agent_id,
                    "source": chunk.provenance.source,
                },
            )
        yield _sse(
            "done",
            {
                "tokens_used": assembled.tokens_used,
                "num_chunks": len(assembled.chunks),
                "sources": assembled.sources,
            },
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("", response_model=ListResponse)
def list_memory(
    request: Request,
    namespace: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=500),
    cursor: str | None = Query(default=None),
    manager: MemoryManager = Depends(get_manager),
) -> ListResponse:
    authorize(request, namespace, "read")
    chunks, next_cursor = manager.list_records(namespace=namespace, limit=limit, cursor=cursor)
    return ListResponse(chunks=[ChunkOut.from_chunk(c) for c in chunks], next_cursor=next_cursor)


@router.delete("", response_model=ForgetResponse)
def forget_memory(
    request: Request,
    namespace: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    manager: MemoryManager = Depends(get_manager),
) -> ForgetResponse:
    """Erase all memory for a namespace and/or session (right-to-be-forgotten)."""
    if not namespace and not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provide a namespace and/or session_id",
        )
    authorize(request, namespace or "*", "write")
    return ForgetResponse(**manager.forget(namespace=namespace, session_id=session_id))


@router.get("/{record_id}", response_model=ChunkOut)
def get_memory(record_id: str, manager: MemoryManager = Depends(get_manager)) -> ChunkOut:
    chunk = manager.get(record_id)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="record not found")
    return ChunkOut.from_chunk(chunk)


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(record_id: str, manager: MemoryManager = Depends(get_manager)) -> None:
    manager.delete([record_id])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_session(
    req: SummarizeRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> SummarizeResponse:
    authorize(request, req.namespace, "write")
    result = manager.summarize_session(
        session_id=req.session_id,
        namespace=req.namespace,
        agent_id=req.agent_id,
        max_sentences=req.max_sentences,
        store_summary=req.store_summary,
    )
    return SummarizeResponse(summary=result["summary"], chunk_ids=result["chunk_ids"])
