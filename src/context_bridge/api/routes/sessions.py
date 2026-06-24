"""Session-level views over episodic / provenance memory."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, status

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import (
    DistillRequest,
    DistillResponse,
    TimelineResponse,
    TurnRequest,
    TurnsResponse,
)
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/timeline", response_model=TimelineResponse)
def session_timeline(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    manager: MemoryManager = Depends(get_manager),
) -> TimelineResponse:
    episodes = manager.timeline(session_id, limit=limit)
    return TimelineResponse(session_id=session_id, episodes=episodes)


@router.post("/{session_id}/turns", status_code=status.HTTP_204_NO_CONTENT)
def add_turn(
    session_id: str,
    req: TurnRequest,
    manager: MemoryManager = Depends(get_manager),
) -> None:
    """Append an ephemeral conversational turn to working memory (cheap, TTL'd)."""
    manager.remember_turn(
        session_id, {"kind": req.kind, "agent_id": req.agent_id, "content": req.content}
    )


@router.get("/{session_id}/turns", response_model=TurnsResponse)
def recent_turns(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    manager: MemoryManager = Depends(get_manager),
) -> TurnsResponse:
    return TurnsResponse(session_id=session_id, turns=manager.recent_turns(session_id, limit=limit))


@router.post("/{session_id}/distill", response_model=DistillResponse)
def distill_session(
    session_id: str,
    req: DistillRequest,
    request: Request,
    manager: MemoryManager = Depends(get_manager),
) -> DistillResponse:
    """Promote the session's most salient turns into durable, cross-session memory."""
    authorize(request, req.namespace, "write")
    result = manager.distill_session(
        session_id=session_id,
        namespace=req.namespace,
        agent_id=req.agent_id,
        max_promote=req.max_promote,
        min_score=req.min_score,
    )
    return DistillResponse(**result)
