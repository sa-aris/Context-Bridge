# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Shared-memory core: governed write path (chunk → embed → dedup/confidence →
  persist → log) and token-budgeted read path (hybrid search → rerank → MMR →
  assemble).
- Three memory tiers: working (in-process/Redis), semantic (Qdrant dense +
  sparse), and episodic/provenance (SQLAlchemy + Alembic).
- FastAPI service under a `/v1` API surface, plus sync and async SDK clients.
- Hybrid retrieval with client-side Reciprocal Rank Fusion, cross-encoder
  reranking (FastEmbed) and MMR diversification.
- Small-to-big chunking with parent documents stored once in SQL.
- TTL decay: query-time filtering plus a `/maintenance/sweep` endpoint and an
  optional background sweeper.
- Optional abstractive summarizer via any OpenAI-compatible endpoint, with an
  extractive fallback.
- Security: opt-in API-key auth (constant-time comparison), namespace-scoped
  keys, and a pluggable rate limiter (in-memory or Redis).
- Batch write and paginated listing endpoints.
- Observability: Prometheus metrics at `/metrics`, request-ID propagation and a
  structured global error handler.
- Packaging: Apache-2.0 license, multi-stage Dockerfile, GitHub Actions CI, `py.typed`.
- External providers: OpenAI / Cohere embedders and the Cohere reranker.
- Streaming recall via Server-Sent Events (`/v1/memory/query/stream`).
- RBAC: per-key namespace globs with read/write permissions (`API_KEY_POLICIES`).
- Observability extras: Prometheus + Grafana stack, OpenTelemetry tracing.
- Helm chart and a runnable token-savings benchmark.
- Right-to-be-forgotten deletion by namespace/session, embedding-dimension
  guard, bounded input sizes, and credential-safe CORS defaults.
- Tag-driven release workflow (GitHub Release + PyPI trusted publishing).
- Cognitive layer (opt-in, behind protocols): reflective consolidation,
  contradiction detection & truth-maintenance, a knowledge-graph layer,
  outcome-feedback re-ranking, and PII/secret redaction on write.
- Collective learning: per-namespace agent reputation profiles, task-outcome
  credit propagation, and procedural memory (playbooks with success tracking).
- Salient distillation: score ephemeral conversational turns and promote only
  the dwelled-upon ones into durable, cross-session memory
  (`/v1/sessions/{id}/turns` + `/distill`).
- Temporal recall: date-aware context (`include_dates`) and `since`/`until`
  time-window filters on query.
