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
    rate_limit_per_minute: int = 0
    rate_limit_backend: str = "memory"  # "memory" or "redis"
    cors_allow_origins: str = "*"  # comma-separated, or "*"

    # Observability
    metrics_enabled: bool = True

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

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    # Vector store
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "context_bridge"

    # Embeddings
    embed_provider: str = "hashing"
    embed_dense_model: str = "BAAI/bge-small-en-v1.5"
    embed_sparse_model: str = "Qdrant/bm25"
    embed_dim: int = 384

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


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
