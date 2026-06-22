"""Wire up OpenTelemetry export and FastAPI instrumentation (opt-in)."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from context_bridge.config import Settings

logger = logging.getLogger("context_bridge")


def setup_tracing(app: FastAPI, settings: Settings) -> None:
    """Configure an OTLP tracer provider and instrument the app.

    No-ops when tracing is disabled. If enabled but the ``otel`` extra is not
    installed, it logs a warning and continues — tracing should never take the
    service down.
    """
    if not settings.tracing_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:  # pragma: no cover - depends on optional extra
        logger.warning("tracing enabled but the 'otel' extra is not installed; skipping")
        return

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint or None)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info(
        "tracing enabled, exporting to %s", settings.otel_exporter_otlp_endpoint or "default"
    )
