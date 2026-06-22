"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from context_bridge.api.deps import build_container
from context_bridge.api.routes import health, maintenance, memory, sessions
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

    app.include_router(health.router)
    app.include_router(memory.router)
    app.include_router(sessions.router)
    app.include_router(maintenance.router)
    return app


app = create_app()
