"""Prometheus metrics and the /metrics exposition endpoint.

Counters and histograms are module-level singletons (registered once against
the default registry) and are incremented from the route handlers — the core
domain layer stays free of any metrics dependency.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

WRITES = Counter("cb_memory_writes_total", "Total memory write requests")
CHUNKS_STORED = Counter("cb_chunks_stored_total", "Chunks persisted to the store")
CHUNKS_DEDUPED = Counter("cb_chunks_deduped_total", "Chunks suppressed as duplicates")

QUERIES = Counter("cb_queries_total", "Total memory query requests")
QUERY_TOKENS = Histogram(
    "cb_query_tokens_used",
    "Tokens in the assembled context per query",
    buckets=(0, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
)
QUERY_CHUNKS = Histogram(
    "cb_query_chunks_returned",
    "Chunks returned per query",
    buckets=(0, 1, 2, 4, 8, 16, 32),
)

SWEEP_DELETED = Counter("cb_sweep_deleted_total", "Expired memories removed by sweeps")
MAINTENANCE_RUNS = Counter("cb_maintenance_runs_total", "Maintenance cycles executed")

EVENTS = Counter("cb_events_total", "Notable memory events emitted", labelnames=("type",))

REQUEST_LATENCY = Histogram(
    "cb_request_latency_seconds",
    "HTTP request latency",
    labelnames=("method", "endpoint", "status"),
)


class MetricsEmitter:
    """An :class:`EventEmitter` that records each event as a Prometheus counter.

    This keeps the core domain layer free of any metrics dependency: events are
    emitted there, and the wiring layer composes this emitter to observe them.
    """

    def emit(self, event_type: str, namespace: str, data: dict) -> None:
        EVENTS.labels(type=event_type).inc()


router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> Response:
    """Expose metrics in the Prometheus text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
