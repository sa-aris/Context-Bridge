"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from context_bridge.api.deps import build_container
from context_bridge.api.routes import health, memory, sessions
from context_bridge.config import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the component graph once at startup, tear down at shutdown."""
    settings: Settings = app.state.settings
    app.state.container = build_container(settings)
    yield
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
    return app


app = create_app()
