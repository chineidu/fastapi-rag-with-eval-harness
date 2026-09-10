"""Qdrant-backed vector store adapter."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from src.app.vector_store.base import BaseVectorStore
from src.schemas.containers import QdrantConfig
from src.schemas.retrieval import Chunk, SearchHit

logger = logging.getLogger(__name__)

_META_POINT_ID = "__index_meta__"
# uuid5 is deterministic: the meta point needs a stable id across runs so
# ensure_collection can read it back for the idempotency check. uuid4 would
# randomize the id each run and force a rebuild every time.
_META_UUID = str(uuid.uuid5(uuid.NAMESPACE_URL, _META_POINT_ID))
_DISTANCE = Distance.COSINE


def _to_point_id(value: str) -> str:
    """Map an arbitrary string key to a Qdrant-compatible UUID point id.

    Uses uuid5 (deterministic) so a chunk maps to the same point id on every
    run, letting re-indexing upsert onto the existing point instead of
    duplicating it.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


class QdrantVectorStore(BaseVectorStore):
    """Self-hosted Qdrant implementation of :class:`BaseVectorStore`."""

    def __init__(self, cfg: QdrantConfig, client: QdrantClient | None = None) -> None:
        """Connect to Qdrant and bind the configured collection.

        Parameters
        ----------
        cfg : QdrantConfig
            Host, port, and collection name.
        client : QdrantClient | None
            Injectable client (e.g. ``QdrantClient(location=":memory:")``
            for tests). Built from ``cfg`` when ``None``.

        """
        self._collection = cfg.collection
        self._client = client or QdrantClient(host=cfg.host, port=cfg.port)

    def _collection_exists(self) -> bool:
        """Return True when the target collection already exists."""
        return self._client.collection_exists(self._collection)

    def _delete_collection(self) -> None:
        """Drop the target collection."""
        self._client.delete_collection(self._collection)

    def _create_collection(self, dim: int) -> None:
        """Create an empty collection sized for ``dim``-dimensional vectors."""
        logger.info("Creating collection %s", self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=dim, distance=_DISTANCE),
        )

    def _load_meta(self) -> dict[str, Any] | None:
        """Read the single sentinel meta point written by a prior run.

        Returns the stored ``{"model_id", "dim"}`` dict, or ``None`` when the
        collection has no sentinel (e.g. it was created elsewhere or is empty).
        This reads only the tiny bookkeeping point, not the collection's chunk
        data; it is used solely by ``ensure_collection`` for the idempotency
        check.
        """
        result = self._client.retrieve(
            collection_name=self._collection,
            ids=[_META_UUID],
            with_payload=True,
            with_vectors=False,
        )
        return result[0].payload if result else None

    def _store_meta(self, model_id: str, dim: int) -> None:
        """Persist the ``{"model_id", "dim"}`` metadata as a sentinel point."""
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=_META_UUID,
                    vector=[0.0] * dim,
                    payload={"is_meta": True, "model_id": model_id, "dim": dim},
                )
            ],
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store each chunk vector with its metadata as the payload."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = [
            PointStruct(
                id=_to_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_path": chunk.doc_path,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        """Return the top-k chunks, excluding the sentinel meta point."""
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            with_payload=True,
            query_filter=Filter(
                must_not=[FieldCondition(key="is_meta", match=MatchValue(value=True))]
            ),
        )
        hits: list[SearchHit] = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=payload.get("chunk_id", ""),
                    doc_path=payload.get("doc_path", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    score=point.score,
                )
            )
        return hits

    def count(self) -> int:
        """Count indexed chunks, excluding the sentinel meta point."""
        return self._client.count(
            collection_name=self._collection,
            count_filter=Filter(
                must_not=[FieldCondition(key="is_meta", match=MatchValue(value=True))]
            ),
        ).count
