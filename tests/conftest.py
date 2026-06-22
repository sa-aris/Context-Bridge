"""Shared test fixtures.

Everything runs hermetically: the deterministic hashing embedder (no model
downloads), Qdrant's in-process ``:memory:`` backend, the identity reranker and
a throwaway SQLite file. No network, no external services.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from context_bridge.api.deps import Container, build_container
from context_bridge.config import Settings
from context_bridge.core.memory.manager import MemoryManager


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        qdrant_url=":memory:",
        qdrant_collection="test_collection",
        embed_provider="hashing",
        embed_dim=128,
        rerank_provider="identity",
        working_provider="memory",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        dedup_threshold=0.95,
        min_confidence=0.0,
        chunk_size_tokens=64,
        chunk_overlap_tokens=8,
        default_top_k=5,
        default_token_budget=256,
        prefetch_limit=20,
        mmr_lambda=0.6,
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    return build_container(settings)


@pytest.fixture
def manager(container: Container) -> MemoryManager:
    return container.manager


@pytest.fixture
def client(settings: Settings) -> Iterator:
    from fastapi.testclient import TestClient

    from context_bridge.api.app import create_app

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
