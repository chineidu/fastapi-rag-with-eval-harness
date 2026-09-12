"""Backend-agnostic vector store contract and shared idempotency policy."""

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from src.schemas.retrieval import Chunk, CollectionInfo, SearchHit

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Backend-agnostic vector store contract for the indexer and retriever."""

    def ensure_collection(
        self, model_id: str, dim: int, *, fingerprint: str, force: bool = False
    ) -> bool:
        """Create, reuse, or rebuild the target collection.

        Idempotent: when the collection already exists with a matching
        ``model_id``, ``dim``, and ``fingerprint``, do nothing. A mismatch
        (or ``force``) rebuilds from scratch.

        Returns
        -------
        bool
            ``True`` when the collection was created or rebuilt, ``False``
            when an existing collection was reused unchanged.

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

    def chunk_count(self) -> int:
        """Return the number of indexed chunks, excluding any meta sentinel."""
        ...

    def describe(self) -> CollectionInfo:
        """Return the collection name, recorded model/dim, and chunk count."""
        ...


class BaseVectorStore(ABC):
    """Shared idempotency policy; backends implement low-level hooks.

    Subclasses implement the six ``_`` hooks plus the three protocol
    methods (``upsert``, ``search``, ``chunk_count``). The idempotent
    ``ensure_collection`` logic and ``describe`` live here so every
    backend behaves the same way.
    """

    def ensure_collection(
        self, model_id: str, dim: int, *, fingerprint: str, force: bool = False
    ) -> bool:
        """Create or reuse the collection, recording model, dim, and fingerprint."""
        # Reuse an existing, matching collection when possible.
        if self._collection_exists():
            if not force:
                meta = self._load_meta()
                if (
                    meta
                    and meta.get("model_id") == model_id
                    and meta.get("dim") == dim
                    and meta.get("corpus_fingerprint") == fingerprint
                ):
                    logger.info(
                        "Collection already indexed with %s/%d; skipping",
                        model_id,
                        dim,
                    )
                    return False
                # Else: metadata mismatch requires a rebuild.
                logger.warning("Collection metadata mismatch; rebuilding")
            else:
                logger.info("Force rebuild of collection")
            self._delete_collection()
        # Create a fresh collection and record its metadata.
        self._create_collection(dim)
        self._store_meta(model_id, dim, fingerprint)
        return True

    def describe(self) -> CollectionInfo:
        """Return collection name, recorded model/dim, and chunk count."""
        # Report a missing collection without touching the backend.
        if not self._collection_exists():
            return CollectionInfo(
                exists=False,
                collection=self._collection_name(),
                model_id=None,
                dim=None,
                chunk_count=0,
            )
        meta = self._load_meta()
        return CollectionInfo(
            exists=True,
            collection=self._collection_name(),
            model_id=meta.get("model_id") if meta else None,
            dim=meta.get("dim") if meta else None,
            chunk_count=self.chunk_count(),
        )

    @abstractmethod
    def _collection_name(self) -> str:
        """Return the target collection name."""
        ...

    @abstractmethod
    def _collection_exists(self) -> bool:
        """Return True when the target collection exists."""
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
        """Return the stored ``{"model_id", "dim", "corpus_fingerprint"}`` dict, or ``None``."""
        ...

    @abstractmethod
    def _store_meta(self, model_id: str, dim: int, fingerprint: str) -> None:
        """Persist the ``{"model_id", "dim", "corpus_fingerprint"}`` metadata."""
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
    def chunk_count(self) -> int:
        """Count indexed chunks, excluding any meta sentinel."""
        ...
