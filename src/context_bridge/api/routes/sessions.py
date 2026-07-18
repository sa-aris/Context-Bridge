"""Session-level views over episodic / provenance memory."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

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


def _working_session_id(namespace: str, session_id: str) -> str:
    """Create an unambiguous tenant-scoped key for ephemeral working memory."""
    return f"{len(namespace)}:{namespace}:{session_id}"


def _auth_enabled(request: Request) -> bool:
    return bool(request.app.state.settings.api_key_set())


def _require_namespace_when_secured(request: Request, namespace: str | None) -> str | None:
    if namespace is None and _auth_enabled(request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="namespace is required when API-key authentication is enabled",
        )
    return namespace


@router.get("/{session_id}/timeline", response_model=TimelineResponse)
def session_timeline(
    session_id: str,
    request: Request,
    namespace: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    manager: MemoryManager = Depends(get_manager),
) -> TimelineResponse:
    namespace = _require_namespace_when_secured(request, namespace)
    if namespace is None:
        # Backward-compatible local/development behavior. Secured deployments
        # must always name a namespace and take the isolated path below.
        episodes = manager.timeline(session_id, limit=limit)
    else:
        authorize(request, namespace, "read")
        episodes = [
            episode
            for episode in manager.timeline(session_id, limit=10_000)
            if episode.get("namespace") == namespace
        ][:limit]
    return TimelineResponse(session_id=session_id, episodes=episodes)


@router.post("/{session_id}/turns", status_code=status.HTTP_204_NO_CONTENT)
def add_turn(
    session_id: str,
    req: TurnRequest,
    request: Request,
    namespace: str | None = Query(default=None),
    manager: MemoryManager = Depends(get_manager),
) -> None:
    """Append an ephemeral conversational turn to working memory."""
    namespace = _require_namespace_when_secured(request, namespace)
    storage_id = session_id
    if namespace is not None:
        authorize(request, namespace, "write")
        storage_id = _working_session_id(namespace, session_id)
    manager.remember_turn(
        storage_id,
        {"kind": req.kind, "agent_id": req.agent_id, "content": req.content},
    )


@router.get("/{session_id}/turns", response_model=TurnsResponse)
def recent_turns(
    session_id: str,
    request: Request,
    namespace: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    manager: MemoryManager = Depends(get_manager),
) -> TurnsResponse:
    namespace = _require_namespace_when_secured(request, namespace)
    storage_id = session_id
    if namespace is not None:
        authorize(request, namespace, "read")
        storage_id = _working_session_id(namespace, session_id)
    turns = manager.recent_turns(storage_id, limit=limit)
    return TurnsResponse(session_id=session_id, turns=turns)


@router.post("/{session_id}/distill", response_model=DistillResponse)
def distill_session(
    session_id: str,
    req: DistillRequest,
    request: Request,
    manager: MemoryManager = Depends(get_manager),
) -> DistillResponse:
    """Promote the session's most salient turns into durable, cross-session memory."""
    authorize(request, req.namespace, "write")
    storage_id = _working_session_id(req.namespace, session_id)
    if not _auth_enabled(request) and not manager.recent_turns(storage_id, limit=1):
        # Legacy local clients posted turns before session namespaces existed.
        storage_id = session_id
    result = manager.distill_session(
        session_id=storage_id,
        namespace=req.namespace,
        agent_id=req.agent_id,
        max_promote=req.max_promote,
        min_score=req.min_score,
    )
    return DistillResponse(**result)
