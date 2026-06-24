"""Collective-learning endpoints: agent reputation, outcomes, procedures."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import (
    AgentsResponse,
    OutcomeRequest,
    OutcomeResponse,
    ProcedureCreate,
    ProcedureCreated,
    ProcedureOutcomeRequest,
    ProceduresResponse,
)
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(tags=["learning"])


@router.get("/agents", response_model=AgentsResponse)
def agent_leaderboard(
    request: Request,
    namespace: str = Query(default="default"),
    limit: int = Query(default=20, ge=1, le=100),
    manager: MemoryManager = Depends(get_manager),
) -> AgentsResponse:
    """Rank agents by accumulated reputation in a namespace."""
    authorize(request, namespace, "read")
    return AgentsResponse(agents=manager.agent_leaderboard(namespace=namespace, limit=limit))


@router.post("/outcomes", response_model=OutcomeResponse)
def record_outcome(
    req: OutcomeRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> OutcomeResponse:
    """Credit a session's memories and agents by its task outcome."""
    authorize(request, req.namespace, "write")
    result = manager.record_outcome(
        session_id=req.session_id,
        namespace=req.namespace,
        success=req.success,
        weight=req.weight,
    )
    return OutcomeResponse(**result)


@router.post("/procedures", response_model=ProcedureCreated)
def create_procedure(
    req: ProcedureCreate, request: Request, manager: MemoryManager = Depends(get_manager)
) -> ProcedureCreated:
    """Store a reusable playbook (procedural memory)."""
    authorize(request, req.namespace, "write")
    proc_id = manager.create_procedure(
        namespace=req.namespace,
        title=req.title,
        steps=req.steps,
        tags=req.tags,
        created_by=req.created_by,
    )
    return ProcedureCreated(id=proc_id or "")


@router.get("/procedures", response_model=ProceduresResponse)
def list_procedures(
    request: Request,
    namespace: str = Query(default="default"),
    query: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    manager: MemoryManager = Depends(get_manager),
) -> ProceduresResponse:
    """List playbooks for a namespace, best-performing first."""
    authorize(request, namespace, "read")
    return ProceduresResponse(
        procedures=manager.list_procedures(namespace=namespace, query=query, limit=limit)
    )


@router.post("/procedures/{procedure_id}/outcome", status_code=status.HTTP_204_NO_CONTENT)
def procedure_outcome(
    procedure_id: str,
    req: ProcedureOutcomeRequest,
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> None:
    """Record whether using a playbook succeeded, so good ones rise."""
    authorize(request, namespace, "write")
    if not manager.record_procedure_outcome(procedure_id, success=req.success):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="procedure not found")
