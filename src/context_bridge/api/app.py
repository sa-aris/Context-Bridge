"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response

from context_bridge.api import metrics
from context_bridge.api.deps import build_container
from context_bridge.api.routes import health, maintenance, memory, sessions
from context_bridge.api.security import RateLimiter, api_key_guard, rate_limit_guard
from context_bridge.config import Settings, get_settings

logger = logging.getLogger("context_bridge")


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
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    if settings.metrics_enabled:
        _install_metrics(app)

    guarded = [Depends(api_key_guard), Depends(rate_limit_guard)]
    app.include_router(health.router)
    app.include_router(memory.router, dependencies=guarded)
    app.include_router(sessions.router, dependencies=guarded)
    app.include_router(maintenance.router, dependencies=guarded)
    return app


def _install_metrics(app: FastAPI) -> None:
    """Mount the /metrics endpoint and a request-latency middleware."""
    app.include_router(metrics.router)

    @app.middleware("http")
    async def _latency_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        metrics.REQUEST_LATENCY.labels(
            method=request.method, endpoint=endpoint, status=str(response.status_code)
        ).observe(time.perf_counter() - start)
        return response


app = create_app()
