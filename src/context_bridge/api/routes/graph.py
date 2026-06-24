"""Knowledge-graph endpoints: traverse entities and relations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import GraphEdgeOut, GraphResponse
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/neighbors", response_model=GraphResponse)
def neighbors(
    request: Request,
    entity: str = Query(..., min_length=1),
    namespace: str = Query(default="default"),
    hops: int = Query(default=1, ge=1, le=4),
    manager: MemoryManager = Depends(get_manager),
) -> GraphResponse:
    """Return relations reachable from ``entity`` within ``hops`` edges."""
    authorize(request, namespace, "read")
    edges = manager.graph_neighbors(namespace=namespace, entity=entity, hops=hops)
    return GraphResponse(entity=entity, edges=[GraphEdgeOut(**e) for e in edges])
