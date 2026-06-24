"""Collaboration-quality endpoint: a single metric for how well agents share."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import QualityResponse
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(tags=["quality"])


@router.get("/quality", response_model=QualityResponse)
def collaboration_quality(
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> QualityResponse:
    """Return a composite 0-100 score of collaboration health for a namespace."""
    authorize(request, namespace, "read")
    return QualityResponse(**manager.collaboration_quality(namespace=namespace))
