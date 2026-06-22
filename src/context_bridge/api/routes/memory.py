"""Memory read/write endpoints — the core agent-facing contract."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from context_bridge.api import metrics
from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import (
    ChunkOut,
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
    WriteRequest,
    WriteResponse,
)
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/write", response_model=WriteResponse)
def write_memory(
    req: WriteRequest, manager: MemoryManager = Depends(get_manager)
) -> WriteResponse:
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


@router.post("/query", response_model=QueryResponse)
def query_memory(
    req: QueryRequest, manager: MemoryManager = Depends(get_manager)
) -> QueryResponse:
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
    req: SummarizeRequest, manager: MemoryManager = Depends(get_manager)
) -> SummarizeResponse:
    result = manager.summarize_session(
        session_id=req.session_id,
        namespace=req.namespace,
        agent_id=req.agent_id,
        max_sentences=req.max_sentences,
        store_summary=req.store_summary,
    )
    return SummarizeResponse(summary=result["summary"], chunk_ids=result["chunk_ids"])
