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
