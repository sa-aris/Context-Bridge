"""Truth-maintenance endpoints: inspect and resolve detected contradictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import AutoResolveResponse, ConflictResolveRequest
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/conflicts", tags=["conflicts"])


@router.get("")
def list_conflicts(
    request: Request,
    namespace: str = Query(default="default"),
    status_filter: str | None = Query(default=None, alias="status"),
    manager: MemoryManager = Depends(get_manager),
) -> dict:
    authorize(request, namespace, "read")
    return {"conflicts": manager.list_conflicts(namespace=namespace, status=status_filter)}


@router.post("/auto-resolve", response_model=AutoResolveResponse)
def auto_resolve(
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> AutoResolveResponse:
    """Auto-close contradictions where the evidence is decisive (belief revision)."""
    authorize(request, namespace, "write")
    settings = request.app.state.settings
    result = manager.auto_resolve_conflicts(
        namespace=namespace, min_gap=settings.auto_resolve_min_gap
    )
    return AutoResolveResponse(**result)


@router.post("/{conflict_id}/resolve", status_code=status.HTTP_204_NO_CONTENT)
def resolve_conflict(
    conflict_id: str,
    req: ConflictResolveRequest,
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> None:
    authorize(request, namespace, "write")
    if not manager.resolve_conflict(conflict_id, winner_id=req.winner_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conflict not found")
