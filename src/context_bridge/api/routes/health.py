"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from context_bridge.api.deps import Container, get_container
from context_bridge.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check — the process is up and serving."""
    return HealthResponse(status="ok", components={})


@router.get("/healthz", response_model=HealthResponse)
def readiness(container: Container = Depends(get_container)) -> HealthResponse:
    """Readiness check — verify backing services are reachable."""
    components: dict[str, str] = {}

    try:
        with container.db.session() as session:
            session.execute(text("SELECT 1"))
        components["database"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on live infra
        components["database"] = f"error: {exc}"

    try:
        container.store.get("readiness-probe")
        components["vector_store"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on live infra
        components["vector_store"] = f"error: {exc}"

    status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=status, components=components)
