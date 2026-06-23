"""Qdrant-backed vector store with hybrid (dense + sparse) retrieval.

Dense and sparse vectors are stored as *named vectors* on a single point, so a
memory is indexed for both semantic and lexical matching at once. Hybrid search
runs the two queries and merges them with Reciprocal Rank Fusion (RRF) on the
client. RRF is rank-based, so it needs no score normalisation and behaves
identically against a remote server or the in-process (``:memory:``) backend.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from context_bridge.core.models import (
    MemoryRecord,
    Provenance,
    RetrievedChunk,
    SparseVector,
    now_ts,
)

DENSE = "dense"
SPARSE = "sparse"
_RRF_K = 60


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], *, k: int = _RRF_K) -> dict[str, float]:
    """Fuse several ranked id lists into a single id -> score map."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, point_id in enumerate(ranked, start=1):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
    return scores


class QdrantStore:
    """Vector store implementation on top of :class:`qdrant_client.QdrantClient`."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        dim: int,
        with_sparse: bool = True,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.with_sparse = with_sparse

    @classmethod
    def from_url(
        cls,
        *,
        url: str,
        collection: str,
        dim: int,
        api_key: str | None = None,
        with_sparse: bool = True,
    ) -> QdrantStore:
        """Build a store, supporting the special ``:memory:`` location."""
        if url == ":memory:":
            client = QdrantClient(location=":memory:")
        else:
            client = QdrantClient(url=url, api_key=api_key or None)
        return cls(client, collection=collection, dim=dim, with_sparse=with_sparse)

    # -- schema -----------------------------------------------------------
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._verify_dim()
            return
        sparse_config = (
            {SPARSE: models.SparseVectorParams(index=models.SparseIndexParams())}
            if self.with_sparse
            else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config=sparse_config,
        )

    def _verify_dim(self) -> None:
        """Fail loudly if an existing collection's dimension won't fit the embedder."""
        vectors = self.client.get_collection(self.collection).config.params.vectors
        existing = None
        if isinstance(vectors, dict):
            params = vectors.get(DENSE)
            existing = getattr(params, "size", None)
        else:
            existing = getattr(vectors, "size", None)
        if existing is not None and existing != self.dim:
            raise ValueError(
                f"collection '{self.collection}' has dense dimension {existing}, but the "
                f"configured embedder produces {self.dim}. Point QDRANT_COLLECTION at a new "
                f"name or recreate the collection after changing the embedding model."
            )

    # -- writes -----------------------------------------------------------
    def upsert(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        points: list[models.PointStruct] = []
        for r in records:
            if r.dense is None:
                raise ValueError(f"record {r.id} is missing its dense vector")
            vector: dict[str, Any] = {DENSE: r.dense}
            if self.with_sparse and r.sparse and not r.sparse.is_empty():
                vector[SPARSE] = models.SparseVector(
                    indices=r.sparse.indices, values=r.sparse.values
                )
            points.append(models.PointStruct(id=r.id, vector=vector, payload=self._to_payload(r)))
        self.client.upsert(collection_name=self.collection, points=points)

    # -- reads ------------------------------------------------------------
    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse: SparseVector | None,
        limit: int,
        namespace: str | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        flt = self._build_filter(namespace, filters)
        prefetch = max(limit * 4, limit)

        dense_ids = [
            str(p.id)
            for p in self.client.query_points(
                collection_name=self.collection,
                query=dense,
                using=DENSE,
                limit=prefetch,
                query_filter=flt,
                with_payload=False,
            ).points
        ]

        ranked_lists = [dense_ids]
        if self.with_sparse and sparse is not None and not sparse.is_empty():
            sparse_ids = [
                str(p.id)
                for p in self.client.query_points(
                    collection_name=self.collection,
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=prefetch,
                    query_filter=flt,
                    with_payload=False,
                ).points
            ]
            ranked_lists.append(sparse_ids)

        fused = _reciprocal_rank_fusion(ranked_lists)
        if not fused:
            return []
        top_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
        return self._hydrate(top_ids, fused)

    def get(self, record_id: str) -> RetrievedChunk | None:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[record_id],
            with_payload=True,
            with_vectors=[DENSE],
        )
        if not records:
            return None
        return self._to_chunk(records[0], score=1.0)

    def delete(self, record_ids: list[str]) -> None:
        if not record_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=list(record_ids)),
        )

    def delete_by(self, *, namespace: str | None = None, session_id: str | None = None) -> int:
        """Delete all points matching a namespace and/or session; return the count."""
        must: list[Any] = []
        if namespace:
            must.append(
                models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))
            )
        if session_id:
            must.append(
                models.FieldCondition(
                    key="provenance.session_id", match=models.MatchValue(value=session_id)
                )
            )
        if not must:
            raise ValueError("delete_by requires a namespace and/or session_id")
        flt = models.Filter(must=must)
        count = self.client.count(self.collection, count_filter=flt, exact=True).count
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt))
        return count

    def sweep_expired(self, *, batch_size: int = 256) -> int:
        """Scroll the collection and delete every record past its TTL."""
        now = now_ts()
        offset: Any = None
        expired: list[str] = []
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                prov = (point.payload or {}).get("provenance", {})
                ttl = prov.get("ttl_seconds")
                if ttl is not None and now - prov.get("created_at", 0.0) > ttl:
                    expired.append(str(point.id))
            if offset is None:
                break
        self.delete(expired)
        return len(expired)

    def list_records(
        self, *, namespace: str | None, limit: int, cursor: str | None
    ) -> tuple[list[RetrievedChunk], str | None]:
        points, next_offset = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=self._build_filter(namespace, None),
            limit=limit,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [self._to_chunk(point, score=0.0) for point in points]
        return chunks, (str(next_offset) if next_offset is not None else None)

    # -- helpers ----------------------------------------------------------
    def _hydrate(self, ids: list[str], scores: dict[str, float]) -> list[RetrievedChunk]:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=list(ids),
            with_payload=True,
            with_vectors=[DENSE],
        )
        by_id = {str(r.id): r for r in records}
        chunks: list[RetrievedChunk] = []
        for point_id in ids:
            record = by_id.get(str(point_id))
            if record is not None:
                chunks.append(self._to_chunk(record, score=scores.get(point_id, 0.0)))
        return chunks

    def _build_filter(self, namespace: str | None, filters: dict | None) -> models.Filter | None:
        must: list[Any] = []
        if namespace:
            must.append(
                models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))
            )
        for key, value in (filters or {}).items():
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=must) if must else None

    @staticmethod
    def _to_payload(r: MemoryRecord) -> dict:
        p = r.provenance
        return {
            "content": r.content,
            "namespace": r.namespace,
            "parent_id": r.parent_id,
            # parent text is intentionally not stored here; it lives once in the
            # parent-document store and is hydrated on demand for expansion.
            "tags": r.tags,
            "metadata": r.metadata,
            "provenance": {
                "agent_id": p.agent_id,
                "session_id": p.session_id,
                "task_id": p.task_id,
                "source": p.source,
                "confidence": p.confidence,
                "created_at": p.created_at,
                "ttl_seconds": p.ttl_seconds,
            },
        }

    @staticmethod
    def _to_chunk(record, *, score: float) -> RetrievedChunk:
        payload = record.payload or {}
        prov = payload.get("provenance", {})
        dense = None
        if record.vector is not None:
            dense = record.vector.get(DENSE) if isinstance(record.vector, dict) else None
        return RetrievedChunk(
            id=str(record.id),
            content=payload.get("content", ""),
            score=score,
            namespace=payload.get("namespace", ""),
            provenance=Provenance(
                agent_id=prov.get("agent_id", "unknown"),
                session_id=prov.get("session_id", "unknown"),
                task_id=prov.get("task_id"),
                source=prov.get("source"),
                confidence=prov.get("confidence", 1.0),
                created_at=prov.get("created_at", 0.0),
                ttl_seconds=prov.get("ttl_seconds"),
            ),
            parent_id=payload.get("parent_id", ""),
            parent_text=payload.get("parent_text"),
            tags=list(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
            dense=list(dense) if dense is not None else None,
        )
