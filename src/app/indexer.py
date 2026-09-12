"""Document indexer: chunk, embed, and upsert into a vector store."""

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from src.app.chunker import DEFAULT_CHUNK_SIZE, chunk_directory
from src.app.vector_store import VectorStore
from src.embeddings.base import AbstractEmbedder
from src.schemas.retrieval import Chunk, IndexReport

logger = logging.getLogger(__name__)


def _corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Hash chunk identities and texts into a stable corpus fingerprint.

    The fingerprint covers the file set, the file contents, and the
    chunking parameters, because all three are encoded in ``chunk_id``
    and ``text``.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Chunks produced by chunking the corpus.

    Returns
    -------
    str
        Hex SHA-256 digest, independent of chunk input order.

    """
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(chunk.text.encode("utf-8")).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


class Indexer:
    """Build a vector index over a corpus directory.

    Orchestrates chunking, embedding, and upsert. The vector store owns
    idempotency (reuse vs rebuild) via a corpus fingerprint, so repeated
    ``build()`` calls on an unchanged corpus skip embedding and upsert.
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

    def build(
        self, corpus_roots: Path | Sequence[Path], *, force: bool = False
    ) -> IndexReport:
        """Chunk, embed, and upsert one or more corpus directories.

        Parameters
        ----------
        corpus_roots : Path | Sequence[Path]
            Directory (or directories) of ``.md``/``.py`` files to index.
        force : bool
            Rebuild the collection even if the corpus fingerprint matches.

        Returns
        -------
        IndexReport
            ``indexed_chunks`` counts chunks embedded and upserted during
            this call; ``skipped`` is ``True`` when the corpus fingerprint
            matched and the index was left untouched.

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
            return IndexReport(indexed_chunks=0, skipped=False)
        # Ensure the collection matches the corpus fingerprint (skip when idempotent).
        model_id = self._embedder.model_id
        dim = self._embedder.dim
        fingerprint = _corpus_fingerprint(chunks)
        rebuilt = self._store.ensure_collection(
            model_id, dim, fingerprint=fingerprint, force=force
        )
        if not rebuilt:
            logger.info(
                "Corpus unchanged; index already has %d chunks",
                self._store.chunk_count(),
            )
            return IndexReport(indexed_chunks=0, skipped=True)
        # Embed and persist in one pass.
        texts: list[str] = [chunk.text for chunk in chunks]
        vectors: list[list[float]] = self._embedder.embed_texts(texts)
        self._store.upsert(chunks, vectors)
        logger.info("Indexed %d chunks (model=%s, dim=%d)", len(chunks), model_id, dim)
        return IndexReport(indexed_chunks=len(chunks), skipped=False)
