"""Console entry point: run the API server with uvicorn."""

from __future__ import annotations

from context_bridge.config import get_settings


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "context_bridge.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
