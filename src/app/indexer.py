"""Document indexer: chunk, embed, and upsert into a vector store."""

import logging
from collections.abc import Sequence
from pathlib import Path

from src.app.chunker import DEFAULT_CHUNK_SIZE, chunk_directory
from src.app.vector_store import VectorStore
from src.embeddings.base import AbstractEmbedder
from src.schemas.retrieval import Chunk

logger = logging.getLogger(__name__)


class Indexer:
    """Build a vector index over a corpus directory.

    Orchestrates chunking, embedding, and upsert. The vector store owns
    idempotency (skip vs rebuild), so repeated ``build()`` calls on an
    unchanged corpus are cheap.
    """

    def __init__(
        self,
        embedder: AbstractEmbedder,
        store: VectorStore,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = 0,
    ) -> None:
        """Configure the indexer.

        Parameters
        ----------
        embedder : AbstractEmbedder
            Embedder used for chunk texts.
        store : VectorStore
            Backend that persists vectors and metadata.
        chunk_size : int
            Characters per chunk (forwarded to the chunker).
        overlap : int
            Characters reused between consecutive chunks.

        """
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._overlap = overlap

    def build(self, corpus_roots: Path | Sequence[Path], *, force: bool = False) -> int:
        """Chunk, embed, and upsert one or more corpus directories.

        Parameters
        ----------
        corpus_roots : Path | Sequence[Path]
            Directory (or directories) of ``.md``/``.py`` files to index.
        force : bool
            Rebuild the collection even if it already matches the embedder.

        Returns
        -------
        int
            Number of chunks indexed.

        Raises
        ------
        TypeError
            If ``corpus_roots`` is a string.
        ValueError
            If ``corpus_roots`` is an empty sequence.

        """
        # Reject bare strings, which would otherwise iterate character by character.
        if isinstance(corpus_roots, str):
            raise TypeError(
                "corpus_roots must be a Path or a sequence of Paths, "
                f"not {type(corpus_roots).__name__}"
            )
        # Normalize to a non-empty tuple of roots.
        roots = (
            (corpus_roots,) if isinstance(corpus_roots, Path) else tuple(corpus_roots)
        )
        if not roots:
            raise ValueError("corpus_roots must not be empty")
        # Chunk every root into one combined document list.
        chunks: list[Chunk] = []
        for root in roots:
            chunks.extend(
                chunk_directory(
                    root, chunk_size=self._chunk_size, overlap=self._overlap
                )
            )
        if not chunks:
            logger.warning("No chunks produced from %s", roots)
            return 0
        # Ensure the collection matches the embedder (skip when idempotent).
        model_id = self._embedder.model_id
        dim = self._embedder.dim
        self._store.ensure_collection(model_id, dim, force=force)
        # Embed and persist in one pass.
        texts: list[str] = [chunk.text for chunk in chunks]
        vectors: list[list[float]] = self._embedder.embed_texts(texts)
        self._store.upsert(chunks, vectors)
        logger.info("Indexed %d chunks (model=%s, dim=%d)", len(chunks), model_id, dim)
        return len(chunks)
