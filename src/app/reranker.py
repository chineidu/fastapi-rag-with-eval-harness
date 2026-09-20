"""Cross-encoder reranking over fused chunk candidates."""

import asyncio
import logging
from typing import Any

from src.schemas.retrieval import SearchHit
from src.schemas.types import DEFAULT_RERANK_MODEL_ID

logger = logging.getLogger(__name__)

__all__ = ["Reranker"]


class Reranker:
    """Rerank chunk hits with a FastEmbed cross-encoder.

    The model loads lazily on first use so construction never downloads
    weights or touches the network. Scores replace RRF scores on the
    returned hits; chunk metadata travels unchanged.
    """

    def __init__(self, model_id: str = DEFAULT_RERANK_MODEL_ID) -> None:
        """Bind the reranker to a model id.

        Parameters
        ----------
        model_id : str
            FastEmbed cross-encoder model name.

        """
        self._model_id = model_id
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        """Return the configured cross-encoder model id."""
        return self._model_id

    def _ensure_model(self) -> Any:
        """Lazily load the FastEmbed cross-encoder."""
        # Import here so module import never downloads weights.
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self._model_id)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        """Return the top_n hits in cross-encoder score order.

        Parameters
        ----------
        query : str
            Raw user query text.
        hits : list[SearchHit]
            Fused chunk candidates in rank order.
        top_n : int
            Maximum hits to return. Must be positive.

        Returns
        -------
        list[SearchHit]
            Top_n hits sorted by cross-encoder relevance, descending.

        Raises
        ------
        ValueError
            If ``top_n`` is not positive.

        """
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        if not hits:
            return []
        # Score each query-chunk pair jointly; the cross-encoder sees
        # token-level interactions instead of separate vectors.
        model = self._ensure_model()
        texts = [hit.text for hit in hits]
        scores = list(model.rerank(query, texts))
        # Pair, sort by cross-encoder relevance, and keep metadata.
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        reranked = [
            SearchHit(
                chunk_id=hit.chunk_id,
                doc_path=hit.doc_path,
                chunk_index=hit.chunk_index,
                text=hit.text,
                score=float(score),
            )
            for hit, score in ranked[:top_n]
        ]
        logger.debug("Reranked %d hits to top %d", len(hits), len(reranked))
        return reranked

    async def arerank(
        self, query: str, hits: list[SearchHit], top_n: int
    ) -> list[SearchHit]:
        """Rerank without blocking the event loop.

        Parameters
        ----------
        query : str
            Raw user query text.
        hits : list[SearchHit]
            Fused chunk candidates in rank order.
        top_n : int
            Maximum hits to return. Must be positive.

        Returns
        -------
        list[SearchHit]
            Top_n hits sorted by cross-encoder relevance, descending.

        """
        return await asyncio.to_thread(self.rerank, query, hits, top_n)
