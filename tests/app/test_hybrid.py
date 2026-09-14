"""Tests for the tantivy lexical index and RRF fusion."""

from pathlib import Path

import pytest

from src.app.hybrid import TantivyIndex, _load_stopwords, _query_tokens, rrf_fuse
from src.schemas.retrieval import Chunk, SearchHit


def _hit(chunk_id: str, doc_path: str, score: float, text: str = "text") -> SearchHit:
    """Build one SearchHit for fusion tests."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_path=doc_path,
        chunk_index=0,
        text=text,
        score=score,
    )


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one Chunk for lexical index tests."""
    return Chunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
        doc_path=chunk_id.split("#")[0],
    )


class TestRrfFuse:
    """Tests for reciprocal rank fusion."""

    def test_shared_chunk_scores_from_both_lists(self) -> None:
        """Given a chunk in both lists, then both rank contributions count."""
        # Given
        dense = [_hit("a#0000", "a.md", 0.9)]
        sparse = [_hit("a#0000", "a.md", 8.0)]
        # When
        fused = rrf_fuse(dense, sparse)
        # Then
        assert len(fused) == 1
        assert fused[0].chunk_id == "a#0000"
        # Default 1:1 weights normalize to 0.5:0.5.
        assert fused[0].score == pytest.approx(1.0 / 61.0)

    def test_weights_are_normalized_to_one(self) -> None:
        """Given weights 3:1, then they behave as 0.75:0.25."""
        # Given
        dense = [_hit("a#0000", "a.md", 0.9), _hit("b#0000", "b.md", 0.8)]
        sparse = [_hit("b#0000", "b.md", 9.0)]
        # When
        fused_31 = rrf_fuse(dense, sparse, dense_weight=3.0, sparse_weight=1.0)
        fused_7525 = rrf_fuse(dense, sparse, dense_weight=0.75, sparse_weight=0.25)
        # Then
        assert [hit.chunk_id for hit in fused_31] == [
            hit.chunk_id for hit in fused_7525
        ]
        assert fused_31[0].score == pytest.approx(fused_7525[0].score)

    def test_ranks_by_fused_score_descending(self) -> None:
        """Given mixed membership, then fused score order wins over raw scores."""
        # Given
        dense = [_hit("a#0000", "a.md", 0.9), _hit("b#0000", "b.md", 0.8)]
        sparse = [_hit("b#0000", "b.md", 9.0)]
        # When
        fused = rrf_fuse(dense, sparse)
        # Then
        assert [hit.chunk_id for hit in fused] == ["b#0000", "a#0000"]

    def test_keeps_first_seen_metadata(self) -> None:
        """Given the same chunk in both lists, then the dense metadata is kept."""
        # Given
        dense = [_hit("a#0000", "a.md", 0.9, text="dense text")]
        sparse = [_hit("a#0000", "a.md", 8.0, text="sparse text")]
        # When
        fused = rrf_fuse(dense, sparse)
        # Then
        assert fused[0].text == "dense text"

    def test_deduplicates_repeated_chunk_in_one_list(self) -> None:
        """Given a chunk repeated in one list, then it appears once in output."""
        # Given
        dense = [_hit("a#0000", "a.md", 0.9), _hit("a#0000", "a.md", 0.8)]
        # When
        fused = rrf_fuse(dense, [])
        # Then
        assert len(fused) == 1

    def test_empty_inputs_return_empty(self) -> None:
        """Given no hits, then fusion returns an empty list."""
        assert rrf_fuse([], []) == []

    def test_rejects_non_positive_rrf_k(self) -> None:
        """Given rrf_k <= 0, then fusion raises ValueError."""
        with pytest.raises(ValueError, match="rrf_k"):
            rrf_fuse([], [], rrf_k=0)

    def test_rejects_non_positive_dense_weight(self) -> None:
        """Given dense_weight <= 0, then fusion raises ValueError."""
        with pytest.raises(ValueError, match="dense_weight"):
            rrf_fuse([], [], dense_weight=0.0)

    def test_rejects_non_positive_sparse_weight(self) -> None:
        """Given sparse_weight <= 0, then fusion raises ValueError."""
        with pytest.raises(ValueError, match="sparse_weight"):
            rrf_fuse([], [], sparse_weight=-1.0)


class TestQueryTokens:
    """Tests for query tokenization."""

    def test_folds_case_and_dedupes(self) -> None:
        """Repeated mixed-case tokens collapse to one lowercase token."""
        # Given / When
        tokens = _query_tokens("Zebra zebra ZEBRA")
        # Then
        assert tokens == ["zebra"]

    def test_removes_stopwords_and_short_tokens(self) -> None:
        """Stopwords and sub-3-char tokens are dropped."""
        # Given / When
        tokens = _query_tokens("how to use the OAuth2 API")
        # Then
        assert "how" not in tokens
        assert "to" not in tokens
        assert "the" not in tokens
        assert "use" in tokens
        assert "oauth2" in tokens
        assert "api" in tokens

    def test_strips_punctuation(self) -> None:
        """Code and punctuation split into alphanumeric runs."""
        # Given / When
        tokens = _query_tokens("OAuth2PasswordRequestForm (422)?")
        # Then
        assert tokens == ["oauth2passwordrequestform", "422"]


class TestLoadStopwords:
    """Tests for the stopword list loader."""

    def test_loads_canonical_words(self) -> None:
        """Common Snowball words load, content words do not."""
        # Given / When
        stopwords = _load_stopwords()
        # Then
        assert "the" in stopwords
        assert "and" in stopwords
        assert "zebra" not in stopwords
        assert len(stopwords) > 100


class TestTantivyIndex:
    """Tests for the persisted lexical index (skipped without tantivy)."""

    @pytest.fixture(autouse=True)
    def _require_tantivy(self) -> None:
        """Skip the whole class when tantivy is not installed."""
        pytest.importorskip("tantivy")

    def test_build_and_search_roundtrip(self, tmp_path: Path) -> None:
        """Given a built index, then a term query returns the matching chunk."""
        # Given
        index = TantivyIndex(tmp_path / "lex")
        index.build([_chunk("a.md#0000", "zebra stripes are unique")])
        # When
        hits = index.search("zebra", 5)
        # Then
        assert len(hits) == 1
        assert hits[0].chunk_id == "a.md#0000"
        assert hits[0].doc_path == "a.md"
        assert "zebra" in hits[0].text

    def test_rebuild_replaces_old_documents(self, tmp_path: Path) -> None:
        """Given a rebuilt index, then stale documents are gone."""
        # Given
        index = TantivyIndex(tmp_path / "lex")
        index.build([_chunk("a.md#0000", "zebra stripes are unique")])
        index.build([_chunk("b.md#0000", "giraffes have long necks")])
        # When
        hits = index.search("zebra", 5)
        # Then
        assert hits == []

    def test_search_missing_index_returns_empty(self, tmp_path: Path) -> None:
        """Given no built index, then search returns no hits."""
        # Given
        index = TantivyIndex(tmp_path / "never_built")
        # When
        hits = index.search("zebra", 5)
        # Then
        assert hits == []

    def test_search_blank_query_returns_empty(self, tmp_path: Path) -> None:
        """Given a blank query, then search returns no hits."""
        # Given
        index = TantivyIndex(tmp_path / "lex")
        index.build([_chunk("a.md#0000", "zebra stripes are unique")])
        # When
        hits = index.search("   ", 5)
        # Then
        assert hits == []

    def test_search_rejects_non_positive_k(self, tmp_path: Path) -> None:
        """Given k <= 0, then search raises ValueError."""
        # Given
        index = TantivyIndex(tmp_path / "lex")
        # When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            index.search("zebra", 0)

    def test_empty_build_clears_stale_index(self, tmp_path: Path) -> None:
        """Given a populated index, then an empty rebuild clears it."""
        # Given
        index = TantivyIndex(tmp_path / "lex")
        index.build([_chunk("a.md#0000", "zebra stripes are unique")])
        # When
        index.build([])
        # Then
        assert index.search("zebra", 5) == []
