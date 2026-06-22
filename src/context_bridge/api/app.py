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

from context_bridge.api import metrics
from context_bridge.api.deps import build_container
from context_bridge.api.routes import health, maintenance, memory, sessions
from context_bridge.api.security import api_key_guard, build_rate_limiter, rate_limit_guard
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the component graph once at startup, tear down at shutdown."""
    settings: Settings = app.state.settings
    app.state.container = build_container(settings)

    sweeper: asyncio.Task | None = None
    if settings.sweep_interval_seconds > 0:
        sweeper = asyncio.create_task(_sweep_loop(app, settings.sweep_interval_seconds))

    try:
        yield
    finally:
        if sweeper is not None:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
        app.state.container = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the Context Bridge API application."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Context Bridge",
        version="0.1.0",
        summary="Shared neural memory middleware for multi-agent systems.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiter = build_rate_limiter(settings)

    _install_middleware(app, settings)
    _install_error_handler(app)

    guarded = [Depends(api_key_guard), Depends(rate_limit_guard)]
    app.include_router(health.router)
    if settings.metrics_enabled:
        app.include_router(metrics.router)
    app.include_router(memory.router, prefix=API_V1, dependencies=guarded)
    app.include_router(sessions.router, prefix=API_V1, dependencies=guarded)
    app.include_router(maintenance.router, prefix=API_V1, dependencies=guarded)
    return app


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
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
