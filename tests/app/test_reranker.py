"""Tests for the FastEmbed cross-encoder reranker."""

import pytest

from src.app.reranker import Reranker
from src.schemas.retrieval import SearchHit
from src.schemas.types import DEFAULT_RERANK_MODEL_ID


def _hit(chunk_id: str, doc_path: str, score: float, text: str = "text") -> SearchHit:
    """Build one SearchHit for reranker tests."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_path=doc_path,
        chunk_index=0,
        text=text,
        score=score,
    )


class FakeCrossEncoder:
    """Cross-encoder stub returning fixed per-document scores."""

    def __init__(self, scores: list[float]) -> None:
        """Store the fixed score list."""
        self._scores = scores

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Return fixed scores aligned with input texts."""
        assert len(texts) == len(self._scores)
        return self._scores


def _reranker_with(scores: list[float]) -> Reranker:
    """Build a Reranker with a stubbed cross-encoder."""
    reranker = Reranker()
    reranker._model = FakeCrossEncoder(scores)
    return reranker


class TestRerankerValidation:
    def test_top_n_must_be_positive(self) -> None:
        """Given non-positive top_n, then rerank raises ValueError."""
        # Given
        reranker = _reranker_with([0.5])
        hits = [_hit("a#0000", "a.md", 0.9)]
        # When / Then
        with pytest.raises(ValueError, match="top_n must be positive"):
            reranker.rerank("query", hits, 0)

    def test_empty_hits_returns_empty_without_model(self) -> None:
        """Given no hits, then rerank returns empty and never loads a model."""
        # Given
        reranker = Reranker()
        # When
        result = reranker.rerank("query", [], 30)
        # Then
        assert result == []
        assert reranker._model is None


class TestRerankerOrdering:
    def test_sorts_by_cross_encoder_score(self) -> None:
        """Given fused hits, then output follows cross-encoder order."""
        # Given
        reranker = _reranker_with([0.1, 0.9, 0.5])
        hits = [
            _hit("a#0000", "a.md", 0.9, "first"),
            _hit("b#0000", "b.md", 0.8, "second"),
            _hit("c#0000", "c.md", 0.7, "third"),
        ]
        # When
        result = reranker.rerank("query", hits, 3)
        # Then
        assert [hit.chunk_id for hit in result] == ["b#0000", "c#0000", "a#0000"]
        assert [hit.score for hit in result] == pytest.approx([0.9, 0.5, 0.1])

    def test_top_n_slices_after_sort(self) -> None:
        """Given top_n smaller than candidates, then only the best survive."""
        # Given
        reranker = _reranker_with([0.1, 0.9, 0.5])
        hits = [
            _hit("a#0000", "a.md", 0.9),
            _hit("b#0000", "b.md", 0.8),
            _hit("c#0000", "c.md", 0.7),
        ]
        # When
        result = reranker.rerank("query", hits, 2)
        # Then
        assert [hit.chunk_id for hit in result] == ["b#0000", "c#0000"]

    def test_preserves_chunk_metadata(self) -> None:
        """Given a reranked hit, then chunk identity and text travel unchanged."""
        # Given
        reranker = _reranker_with([0.9])
        hits = [_hit("a#0000", "a.md", 0.1, text="body text")]
        # When
        result = reranker.rerank("query", hits, 1)
        # Then
        assert result[0].chunk_id == "a#0000"
        assert result[0].doc_path == "a.md"
        assert result[0].text == "body text"


class TestRerankerLazy:
    def test_construction_never_loads_model(self) -> None:
        """Given a new Reranker, then no model is loaded until rerank."""
        # Given / When
        reranker = Reranker(model_id=DEFAULT_RERANK_MODEL_ID)
        # Then
        assert reranker.model_id == DEFAULT_RERANK_MODEL_ID
        assert reranker._model is None


class TestARerank:
    async def test_arerank_delegates_to_rerank(self) -> None:
        """Given hits, then arerank returns the same order as rerank."""
        # Given
        reranker = _reranker_with([0.2, 0.8])
        hits = [_hit("a#0000", "a.md", 0.9), _hit("b#0000", "b.md", 0.8)]
        # When
        result = await reranker.arerank("query", hits, 2)
        # Then
        assert [hit.chunk_id for hit in result] == ["b#0000", "a#0000"]
