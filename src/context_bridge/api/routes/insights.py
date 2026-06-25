"""Insight endpoints: namespace health panel and belief timeline (memory diff)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import BeliefTimelineResponse, HealthPanelResponse
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(tags=["insights"])


@router.get("/namespaces/{namespace}/health", response_model=HealthPanelResponse)
def namespace_health(
    namespace: str,
    request: Request,
    manager: MemoryManager = Depends(get_manager),
) -> HealthPanelResponse:
    """A single pulse-check: volume, trust distribution, conflicts, lessons, quality."""
    authorize(request, namespace, "read")
    return HealthPanelResponse(**manager.namespace_health(namespace=namespace))


@router.get("/namespaces/{namespace}/beliefs", response_model=BeliefTimelineResponse)
def belief_timeline(
    namespace: str,
    request: Request,
    query: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    manager: MemoryManager = Depends(get_manager),
) -> BeliefTimelineResponse:
    """Trace how belief about a topic evolved over time (a memory diff)."""
    authorize(request, namespace, "read")
    events = manager.belief_timeline(query=query, namespace=namespace, limit=limit)
    return BeliefTimelineResponse(query=query, events=events)
