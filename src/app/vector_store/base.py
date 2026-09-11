"""Backend-agnostic vector store contract and shared idempotency policy."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from src.schemas.retrieval import Chunk, SearchHit

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Backend-agnostic vector store contract for the indexer and retriever."""

    def ensure_collection(
        self, model_id: str, dim: int, *, force: bool = False
    ) -> None:
        """Create or reuse the target collection.

        Idempotent: when the collection already exists with a matching
        ``model_id`` and ``dim``, do nothing. A mismatch (or ``force``)
        rebuilds from scratch.
        """
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunk vectors with their metadata payloads."""
        ...

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        """Return the top-k most similar chunks with metadata.

        Hits must be ordered by similarity score, descending.
        """
        ...

    def count(self) -> int:
        """Return the number of indexed (non-meta) points."""
        ...


class BaseVectorStore(ABC):
    """Shared idempotency policy; backends implement low-level hooks.

    Subclasses implement the five ``_`` hooks plus the three protocol
    methods (``upsert``, ``search``, ``count``). The idempotent
    ``ensure_collection`` logic lives here so every backend behaves the
    same way.
    """

    def ensure_collection(
        self, model_id: str, dim: int, *, force: bool = False
    ) -> None:
        """Create or reuse the collection, recording model_id and dim."""
        # Reuse an existing, matching collection when possible.
        if self._collection_exists():
            if not force:
                meta = self._load_meta()
                if meta and meta.get("model_id") == model_id and meta.get("dim") == dim:
                    logger.info(
                        "Collection already indexed with %s/%d; skipping",
                        model_id,
                        dim,
                    )
                    return
                # Else: mismatch or force rebuild.
                logger.warning("Collection model/dim mismatch; rebuilding")
            else:
                logger.info("Force rebuild of collection")
            self._delete_collection()
        # Create a fresh collection and record its metadata.
        self._create_collection(dim)
        self._store_meta(model_id, dim)

    @abstractmethod
    def _collection_exists(self) -> bool:
        """Return True when the target collection already exists."""
        ...

    @abstractmethod
    def _delete_collection(self) -> None:
        """Drop the target collection."""
        ...

    @abstractmethod
    def _create_collection(self, dim: int) -> None:
        """Create an empty collection sized for ``dim``-dimensional vectors."""
        ...

    @abstractmethod
    def _load_meta(self) -> dict | None:
        """Return the stored ``{"model_id", "dim"}`` dict, or ``None``."""
        ...

    @abstractmethod
    def _store_meta(self, model_id: str, dim: int) -> None:
        """Persist the ``{"model_id", "dim"}`` metadata for the collection."""
        ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store each chunk vector with its metadata as the payload."""
        ...

    @abstractmethod
    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        """Return the top-k chunks in descending score order, excluding any meta sentinel."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Count indexed chunks, excluding any meta sentinel."""
        ...
