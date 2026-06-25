"""Application configuration loaded from environment / .env files."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, typed configuration for every component.

    Values are read from environment variables (or a local ``.env`` file).
    See ``.env.example`` for the full set of supported keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    # Security (empty api_keys => auth disabled; 0 rate => limiter disabled)
    api_keys: str = ""
    api_key_namespaces: str = ""  # JSON: {"key": ["ns-a", "ns-b"]}; "*" = all
    # JSON RBAC: {"key": {"namespaces": ["team-a*"], "permissions": ["read"]}}
    api_key_policies: str = ""
    rate_limit_per_minute: int = 0
    rate_limit_backend: str = "memory"  # "memory" or "redis"
    cors_allow_origins: str = "*"  # comma-separated, or "*"

    # Observability
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otel_service_name: str = "context-bridge"
    otel_exporter_otlp_endpoint: str = ""  # e.g. http://localhost:4317

    def api_key_set(self) -> set[str]:
        """Parse the comma-separated ``api_keys`` setting into a set."""
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}

    def api_key_namespace_map(self) -> dict[str, list[str]]:
        """Parse the per-key namespace allow-list (empty => all keys get '*')."""
        if not self.api_key_namespaces.strip():
            return {}
        import json

        data = json.loads(self.api_key_namespaces)
        return {str(k): list(v) for k, v in data.items()}

    def api_key_policies_map(self) -> dict[str, dict]:
        """Parse the per-key RBAC policy map (namespaces + permissions)."""
        if not self.api_key_policies.strip():
            return {}
        import json

        data = json.loads(self.api_key_policies)
        return {str(k): dict(v) for k, v in data.items()}

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    # Vector store
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "context_bridge"

    # Embeddings
    embed_provider: str = "hashing"  # hashing | fastembed | openai | cohere
    embed_dense_model: str = "BAAI/bge-small-en-v1.5"
    embed_sparse_model: str = "Qdrant/bm25"
    embed_dim: int = 384

    # External provider credentials (OpenAI-compatible / Cohere)
    openai_api_key: str = ""
    openai_base_url: str = ""
    cohere_api_key: str = ""

    # Reranker
    rerank_provider: str = "identity"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Summarizer ("extractive" needs no model; "llm" calls an
    # OpenAI-compatible chat endpoint and falls back to extractive on error)
    summarizer_provider: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5"
    llm_timeout: float = 30.0
    llm_max_tokens: int = 512

    # Structured store
    database_url: str = "sqlite+pysqlite:///./context_bridge.db"

    # Working memory
    working_provider: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    working_ttl_seconds: int = 3600

    # Retrieval defaults
    default_top_k: int = 8
    default_token_budget: int = 2048
    prefetch_limit: int = 50
    mmr_lambda: float = 0.6

    # Write policy
    dedup_threshold: float = 0.95
    min_confidence: float = 0.0
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32

    # Maintenance — background TTL sweep (0 disables the periodic job)
    sweep_interval_seconds: int = 0

    # Cognitive layer
    redact_pii: bool = False
    feedback_weight: float = 0.15  # how strongly outcome feedback re-ranks recall
    consolidation_min_cluster: int = 3
    consolidation_similarity: float = 0.83
    detect_contradictions: bool = False
    contradiction_similarity: float = 0.80
    graph_extraction: bool = False

    # Failure memory — proactively surface lessons from past mistakes on recall
    lessons_enabled: bool = True
    lessons_top_k: int = 3
    lessons_min_score: float = 0.20  # min trigger↔query similarity to raise a lesson
    # Auto-distil lessons by clustering memories implicated in failures
    lesson_distill_min_cluster: int = 2
    lesson_distill_similarity: float = 0.83

    # Belief revision — demote memories that lose a contradiction, decay their trust
    belief_revision: bool = True
    conflict_loser_decay: float = 0.5  # multiply the loser's confidence on resolve
    confidence_weight: float = 0.5  # how strongly recall demotes low-confidence memories
    auto_resolve_min_gap: float = 0.3  # authority gap needed to auto-close a conflict


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
