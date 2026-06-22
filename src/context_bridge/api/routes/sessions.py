"""Session-level views over episodic / provenance memory."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import TimelineResponse
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
