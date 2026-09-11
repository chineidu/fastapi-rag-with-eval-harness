"""Baseline retrieval adapter: embed a query, search chunks, dedupe to documents."""

import logging

from src.app.vector_store import VectorStore, get_vector_store
from src.config import load_app_config
from src.embeddings import AbstractEmbedder, get_embedder
from src.schemas.harness import RetrievalResult, RetrievedDocument
from src.schemas.models import AppConfig
from src.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)


class LocalRetriever:
    """Baseline retriever adapter over the local vector index.

    The harness asks for documents while the index stores chunks, so the
    retriever over-fetches ``k * overfetch_factor`` chunks and keeps the best
    hit per document (ADR-0021). The embedder and vector store are resolved
    from the shared app config, so query vectors always match the indexed
    vectors.
    """

    def __init__(
        self,
        embedder: AbstractEmbedder | None = None,
        store: VectorStore | None = None,
        *,
        config: AppConfig | None = None,
        overfetch_factor: int | None = None,
    ) -> None:
        """Configure the retriever.

        Parameters
        ----------
        embedder : AbstractEmbedder | None
            Injectable embedder (tests). Defaults to the configured provider.
        store : VectorStore | None
            Injectable vector store (tests). Defaults to the configured backend.
        config : AppConfig | None
            Application config. Loaded from the bundled YAML when ``None``.
        overfetch_factor : int | None
            Chunk-window multiplier. Defaults to
            ``config.retriever_config.overfetch_factor``.

        Raises
        ------
        ValueError
            If the resolved ``overfetch_factor`` is less than 1.

        """
        cfg = config or load_app_config()
        self._embedder = embedder or get_embedder(cfg.embeddings_config)
        self._store = store or get_vector_store(cfg.indexer_config)
        resolved_factor = (
            overfetch_factor
            if overfetch_factor is not None
            else cfg.retriever_config.overfetch_factor
        )
        if resolved_factor < 1:
            raise ValueError(f"overfetch_factor must be >= 1, got {resolved_factor}")
        self._overfetch_factor = resolved_factor
        self._chunk_size = cfg.indexer_config.chunk_size
        self._overlap = cfg.indexer_config.overlap

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Return the top-k documents for a query.

        Parameters
        ----------
        query : str
            The ``query_text`` from a ground truth record.
        k : int
            Number of documents to return. Must be positive.

        Returns
        -------
        RetrievalResult
            Ranked ``RetrievedDocument`` list plus retrieval metadata.

        Raises
        ------
        ValueError
            If ``k`` is not positive.

        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # Embed the query with the same model used at indexing time.
        vector: list[float] = self._embedder.embed_texts([query])[0]
        # Over-fetch chunks so dedupe can still fill k document slots (ADR-0021).
        limit: int = k * self._overfetch_factor
        hits: list[SearchHit] = self._store.search(vector, limit)
        # Rank by score so the first hit per document is its best chunk,
        # regardless of the order the backend returned.
        hits = sorted(hits, key=lambda hit: hit.score, reverse=True)
        # Keep the best (first) hit per document, preserving score order.
        documents: list[RetrievedDocument] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.doc_path in seen:
                continue
            seen.add(hit.doc_path)
            documents.append(
                RetrievedDocument(doc_path=hit.doc_path, score=float(hit.score))
            )
            if len(documents) == k:
                break
        metadata: dict[str, object] = {
            "model_id": self._embedder.model_id,
            "overfetch_factor": self._overfetch_factor,
            "chunk_limit": limit,
            "chunks_fetched": len(hits),
            "docs_returned": len(documents),
            "k": k,
            "chunk_size": self._chunk_size,
            "overlap": self._overlap,
        }
        logger.debug(
            "Retrieved %d docs from %d chunks for k=%d",
            len(documents),
            len(hits),
            k,
        )
        return RetrievalResult(documents=documents, metadata=metadata)

    def generate(self, query: str, documents: list[str]) -> str:
        """Generate an answer from retrieved documents (deferred).

        Parameters
        ----------
        query : str
            The user question.
        documents : list[str]
            Retrieved document paths used as context.

        Returns
        -------
        str
            Never returned; generation eval is out of scope.

        Raises
        ------
        NotImplementedError
            Always. Generation eval is deferred per ADR-0004.

        """
        raise NotImplementedError(
            "generate() is deferred until generation eval is in scope (ADR-0004)"
        )
