"""Retrieval adapter: embed a query, search chunks, dedupe to documents."""

import logging
from pathlib import Path

from src.app.generator import RAGGenerator
from src.app.hybrid import TantivyIndex, rrf_fuse
from src.app.vector_store import VectorStore, get_vector_store
from src.config import load_app_config
from src.embeddings import AbstractEmbedder, get_embedder
from src.schemas.generation import GeneratedAnswer
from src.schemas.harness import RetrievalResult, RetrievedDocument
from src.schemas.models import AppConfig
from src.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)


class LocalRetriever:
    """Retriever adapter over the local dense and lexical indexes.

    The harness asks for documents while the indexes store chunks, so the
    retriever over-fetches ``k * overfetch_factor`` dense chunks and keeps
    the best hit per document (ADR-0021). When hybrid retrieval is enabled,
    ``sparse_k`` BM25 chunk hits from the Tantivy index are fused with the
    dense hits via RRF (``rrf_k``) before dedupe. The embedder and vector
    store are resolved from the shared app config, so query vectors always
    match the indexed vectors.
    """

    def __init__(
        self,
        embedder: AbstractEmbedder | None = None,
        store: VectorStore | None = None,
        *,
        config: AppConfig | None = None,
        overfetch_factor: int | None = None,
        lexical: TantivyIndex | None = None,
        generator: RAGGenerator | None = None,
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
        lexical : TantivyIndex | None
            Injectable lexical index (tests). Built from
            ``config.retriever_config.tantivy_index_dir`` when ``None`` and
            hybrid retrieval is enabled.
        generator : RAGGenerator | None
            Injectable generator (tests). Built lazily from ``config`` on
            first ``agenerate`` call when ``None``, so retrieval-only use
            never constructs an LLM client.

        Raises
        ------
        ValueError
            If the resolved ``overfetch_factor`` is less than 1.

        """
        cfg = config or load_app_config()
        self._config = cfg
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
        self._hybrid_enabled = cfg.retriever_config.hybrid_enabled
        self._sparse_k = cfg.retriever_config.sparse_k
        self._rrf_k = cfg.retriever_config.rrf_k
        self._dense_weight = cfg.retriever_config.dense_weight
        self._sparse_weight = cfg.retriever_config.sparse_weight
        if lexical is not None:
            self._lexical = lexical
        elif self._hybrid_enabled:
            self._lexical = TantivyIndex(Path(cfg.retriever_config.tantivy_index_dir))
        else:
            self._lexical = None
        self._generator = generator

    def _fused_hits(self, query: str, k: int) -> tuple[list[SearchHit], int]:
        """Embed, search, and fuse dense and sparse chunk hits.

        Parameters
        ----------
        query : str
            The ``query_text`` from a ground truth record.
        k : int
            Number of documents wanted; sizes the chunk candidate window.

        Returns
        -------
        tuple[list[SearchHit], int]
            Fused chunk hits in score order plus the sparse fetch count.

        """
        # Embed the query with the same model used at indexing time.
        vector: list[float] = self._embedder.embed_texts([query])[0]
        # Over-fetch chunks so dedupe can still fill k document slots (ADR-0021).
        limit: int = k * self._overfetch_factor
        dense_hits: list[SearchHit] = self._store.search(vector, limit)
        # Rank by score so the first hit per document is its best chunk,
        # regardless of the order the backend returned.
        dense_hits = sorted(dense_hits, key=lambda hit: hit.score, reverse=True)
        sparse_fetched_count = 0
        # Fuse BM25 candidates when hybrid retrieval is enabled.
        if self._hybrid_enabled and self._lexical is not None:
            sparse_hits: list[SearchHit] = self._lexical.search(query, self._sparse_k)
            sparse_fetched_count = len(sparse_hits)
            fused_hits: list[SearchHit] = rrf_fuse(
                dense_hits,
                sparse_hits,
                rrf_k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
            )
        else:
            if self._hybrid_enabled:
                logger.warning(
                    "Hybrid enabled but no lexical index; using dense hits only"
                )
            fused_hits = dense_hits
        return fused_hits, sparse_fetched_count

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
        fused_hits, sparse_fetched_count = self._fused_hits(query, k)
        limit = k * self._overfetch_factor
        # Keep the best (first) hit per document, preserving score order.
        documents: list[RetrievedDocument] = []
        seen: set[str] = set()
        for hit in fused_hits:
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
            "chunks_fetched": len(fused_hits),
            "docs_returned": len(documents),
            "k": k,
            "chunk_size": self._chunk_size,
            "overlap": self._overlap,
            "hybrid_enabled": self._hybrid_enabled,
            "sparse_k": self._sparse_k,
            "rrf_k": self._rrf_k,
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "sparse_fetched_count": sparse_fetched_count,
        }
        logger.debug(
            "Retrieved %d docs from %d chunks for k=%d",
            len(documents),
            len(fused_hits),
            k,
        )
        return RetrievalResult(documents=documents, metadata=metadata)

    async def agenerate(
        self,
        query: str,
        documents: list[SearchHit] | None = None,
        k: int = 10,
    ) -> GeneratedAnswer:
        """Generate a structured answer, retrieving first when needed.

        Parameters
        ----------
        query : str
            The user question.
        documents : list[SearchHit] | None
            Ranked chunk hits supplying grounding. When ``None``,
            retrieve ``k`` documents first and use their best chunks.
        k : int
            Documents to retrieve when ``documents`` is ``None``.

        Returns
        -------
        GeneratedAnswer
            Answer text with citations clamped to context paths.

        Raises
        ------
        ValueError
            If ``documents`` is ``None`` and ``k`` is not positive.

        """
        if documents is not None:
            contexts: list[SearchHit] = documents
        else:
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            fused_hits, _ = self._fused_hits(query, k)
            contexts = []
            seen: set[str] = set()
            for hit in fused_hits:
                if hit.doc_path in seen:
                    continue
                seen.add(hit.doc_path)
                contexts.append(hit)
                if len(contexts) == k:
                    break
        if self._generator is None:
            self._generator = RAGGenerator(config=self._config)
        return await self._generator.agenerate(query, contexts)
