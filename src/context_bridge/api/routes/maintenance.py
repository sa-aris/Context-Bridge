"""Maintenance endpoints (TTL sweep and other housekeeping)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from context_bridge.api import metrics
from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import ConsolidateResponse, SweepResponse
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/maintenance", tags=["maintenance"])


@router.post("/sweep", response_model=SweepResponse)
def sweep_expired(request: Request, manager: MemoryManager = Depends(get_manager)) -> SweepResponse:
    """Physically delete TTL-expired memories from the semantic store.

    A global maintenance action: requires write access across all namespaces.
    """
    authorize(request, "*", "write")
    deleted = manager.sweep_expired()
    metrics.SWEEP_DELETED.inc(deleted)
    return SweepResponse(deleted=deleted)


@router.post("/consolidate", response_model=ConsolidateResponse)
def consolidate(
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> ConsolidateResponse:
    """Cluster a namespace's memories and write back synthesized insights."""
    authorize(request, namespace, "write")
    settings = request.app.state.settings
    result = manager.consolidate(
        namespace=namespace,
        min_cluster=settings.consolidation_min_cluster,
        similarity=settings.consolidation_similarity,
    )
    return ConsolidateResponse(**result)
