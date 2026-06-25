"""Failure-memory endpoints: capture, list and apply lessons from mistakes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from context_bridge.api.deps import get_manager
from context_bridge.api.schemas import (
    LessonCreated,
    LessonRequest,
    LessonsResponse,
    PreflightRequest,
    PreflightResponse,
)
from context_bridge.api.security import authorize
from context_bridge.core.memory.manager import MemoryManager

router = APIRouter(tags=["lessons"])


@router.post("/lessons", response_model=LessonCreated)
def record_lesson(
    req: LessonRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> LessonCreated:
    """Capture a lesson so a past mistake can be flagged before it recurs."""
    authorize(request, req.namespace, "write")
    lesson_id = manager.record_lesson(
        namespace=req.namespace,
        trigger=req.trigger,
        guidance=req.guidance,
        severity=req.severity,
        created_by=req.created_by,
        session_id=req.session_id,
    )
    return LessonCreated(id=lesson_id or "")


@router.get("/lessons", response_model=LessonsResponse)
def list_lessons(
    request: Request,
    namespace: str = Query(default="default"),
    limit: int = Query(default=100, ge=1, le=500),
    manager: MemoryManager = Depends(get_manager),
) -> LessonsResponse:
    """List a namespace's lessons, most-proven first."""
    authorize(request, namespace, "read")
    return LessonsResponse(lessons=manager.list_lessons(namespace=namespace, limit=limit))


@router.post("/lessons/{lesson_id}/confirm", status_code=status.HTTP_204_NO_CONTENT)
def confirm_lesson(
    lesson_id: str,
    request: Request,
    namespace: str = Query(default="default"),
    manager: MemoryManager = Depends(get_manager),
) -> None:
    """Record that a surfaced lesson actually helped, so it ranks higher."""
    authorize(request, namespace, "write")
    if not manager.confirm_lesson(lesson_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="lesson not found")


@router.post("/preflight", response_model=PreflightResponse)
def preflight(
    req: PreflightRequest, request: Request, manager: MemoryManager = Depends(get_manager)
) -> PreflightResponse:
    """Brief an agent on what the collective knows before it starts a task."""
    authorize(request, req.namespace, "read")
    result = manager.preflight(task=req.task, namespace=req.namespace, limit=req.limit)
    return PreflightResponse(**result)
