"""Dependency wiring: build the component graph from settings."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from context_bridge.config import Settings
from context_bridge.core.chunking import build_chunker
from context_bridge.core.embeddings import build_embedder
from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.memory.manager import MemoryManager
from context_bridge.core.memory.policy import WritePolicy
from context_bridge.core.retrieval import Retriever, build_reranker
from context_bridge.core.retrieval.retriever import RetrievalParams
from context_bridge.core.vectorstore import build_vector_store
from context_bridge.core.vectorstore.base import VectorStore
from context_bridge.core.working import build_working_memory
from context_bridge.db import Database, EpisodeRepository


@dataclass(slots=True)
class Container:
    """Holds the constructed singletons for the lifetime of the process."""

    settings: Settings
    embedder: Embedder
    store: VectorStore
    db: Database
    manager: MemoryManager


def build_container(settings: Settings) -> Container:
    """Construct every component from configuration and wire the manager."""
    embedder = build_embedder(settings)
    store = build_vector_store(
        settings, dim=embedder.dense_dim, supports_sparse=embedder.supports_sparse
    )
    reranker = build_reranker(settings)
    retriever = Retriever(embedder=embedder, store=store, reranker=reranker)

    working = build_working_memory(settings)
    db = Database(settings.database_url)
    db.create_all()
    episodes = EpisodeRepository(db)

    policy = WritePolicy(
        dedup_threshold=settings.dedup_threshold,
        min_confidence=settings.min_confidence,
    )
    defaults = RetrievalParams(
        top_k=settings.default_top_k,
        token_budget=settings.default_token_budget,
        candidate_pool=settings.prefetch_limit,
        mmr_lambda=settings.mmr_lambda,
    )
    manager = MemoryManager(
        chunker=build_chunker(settings, embedder=embedder),
        embedder=embedder,
        store=store,
        retriever=retriever,
        working=working,
        episodes=episodes,
        policy=policy,
        defaults=defaults,
    )
    return Container(settings=settings, embedder=embedder, store=store, db=db, manager=manager)


def get_container(request: Request) -> Container:
    """FastAPI dependency: the process-wide container from app state."""
    return request.app.state.container


def get_manager(request: Request) -> MemoryManager:
    """FastAPI dependency: the shared :class:`MemoryManager`."""
    return request.app.state.container.manager
