"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from context_bridge import __version__
from context_bridge.api import metrics
from context_bridge.api.access import AccessControl
from context_bridge.api.deps import build_container
from context_bridge.api.routes import (
    conflicts,
    graph,
    health,
    insights,
    learning,
    lessons,
    maintenance,
    memory,
    quality,
    sessions,
)
from context_bridge.api.security import api_key_guard, build_rate_limiter, rate_limit_guard
from context_bridge.api.tracing import setup_tracing
from context_bridge.config import Settings, get_settings

logger = logging.getLogger("context_bridge")

API_V1 = "/v1"
_REQUEST_ID_HEADER = "X-Request-ID"


async def _sweep_loop(app: FastAPI, interval: int) -> None:
    """Periodically purge TTL-expired memories until cancelled."""
    while True:
        await asyncio.sleep(interval)
        try:
            deleted = await asyncio.to_thread(app.state.container.manager.sweep_expired)
            if deleted:
                logger.info("ttl sweep removed %d expired memories", deleted)
        except Exception:  # pragma: no cover - defensive background task
            logger.exception("ttl sweep failed")


async def _maintenance_loop(app: FastAPI, interval: int) -> None:
    """Periodically run a full housekeeping cycle until cancelled."""
    from context_bridge.api.routes.maintenance import _run_maintenance

    while True:
        await asyncio.sleep(interval)
        try:
            settings = app.state.settings
            result = await asyncio.to_thread(
                _run_maintenance, app.state.container.manager, settings
            )
            logger.info("maintenance cycle: %s", result)
        except Exception:  # pragma: no cover - defensive background task
            logger.exception("maintenance cycle failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the component graph once at startup, tear down at shutdown."""
    settings: Settings = app.state.settings
    app.state.container = build_container(settings)

    tasks: list[asyncio.Task] = []
    if settings.sweep_interval_seconds > 0:
        tasks.append(asyncio.create_task(_sweep_loop(app, settings.sweep_interval_seconds)))
    if settings.maintenance_interval_seconds > 0:
        tasks.append(
            asyncio.create_task(_maintenance_loop(app, settings.maintenance_interval_seconds))
        )

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        app.state.container = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the Context Bridge API application."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Context Bridge",
        version=__version__,
        summary="Shared neural memory middleware for multi-agent systems.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiter = build_rate_limiter(settings)
    app.state.access = AccessControl.build(settings)

    setup_tracing(app, settings)
    _install_middleware(app, settings)
    _install_error_handler(app)

    @app.get("/", tags=["meta"])
    def root() -> dict:
        """Service banner with links to docs, health and metrics."""
        return {
            "name": "Context Bridge",
            "description": "Shared neural memory middleware for multi-agent systems.",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics" if settings.metrics_enabled else None,
            "api": API_V1,
        }

    guarded = [Depends(api_key_guard), Depends(rate_limit_guard)]
    app.include_router(health.router)
    if settings.metrics_enabled:
        app.include_router(metrics.router)
    app.include_router(memory.router, prefix=API_V1, dependencies=guarded)
    app.include_router(sessions.router, prefix=API_V1, dependencies=guarded)
    app.include_router(maintenance.router, prefix=API_V1, dependencies=guarded)
    app.include_router(conflicts.router, prefix=API_V1, dependencies=guarded)
    app.include_router(graph.router, prefix=API_V1, dependencies=guarded)
    app.include_router(learning.router, prefix=API_V1, dependencies=guarded)
    app.include_router(lessons.router, prefix=API_V1, dependencies=guarded)
    app.include_router(quality.router, prefix=API_V1, dependencies=guarded)
    app.include_router(insights.router, prefix=API_V1, dependencies=guarded)
    return app


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    origins = settings.cors_origin_list()
    # Browsers reject (and it is unsafe to send) credentials with a "*" origin,
    # so only enable credentialed CORS when explicit origins are configured.
    allow_all = origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _observability(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers[_REQUEST_ID_HEADER] = request_id
        if settings.metrics_enabled:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", request.url.path)
            metrics.REQUEST_LATENCY.labels(
                method=request.method, endpoint=endpoint, status=str(response.status_code)
            ).observe(elapsed)
        return response


def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error (request_id=%s)", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {"type": "internal_error", "message": "internal server error"},
                "request_id": request_id,
            },
        )


app = create_app()
