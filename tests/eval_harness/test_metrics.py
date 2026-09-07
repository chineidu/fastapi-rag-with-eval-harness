"""Tests for the eval_harness.metrics module."""

import pytest

from src.eval_harness.metrics import (
    OVERALL_CATEGORY,
    _aggregate_scores,
    aggregate_by_category,
    precision_at_k,
    recall_at_k,
)
from src.schemas.harness import ScoredQuery


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """Given all relevant docs in top k, then recall is 1.0."""
        # Given
        retrieved = ["a.md", "b.md", "c.md"]
        relevant = ["a.md", "b.md"]

        # When
        result = recall_at_k(retrieved, relevant, k=3)

        # Then
        assert result == 1.0

    def test_partial_recall(self) -> None:
        """Given some relevant docs in top k, then recall is fractional."""
        # Given
        retrieved = ["a.md", "x.md", "y.md"]
        relevant = ["a.md", "b.md"]

        # When
        result = recall_at_k(retrieved, relevant, k=3)

        # Then
        assert result == 0.5

    def test_zero_recall(self) -> None:
        """Given no relevant docs in top k, then recall is 0.0."""
        # Given
        retrieved = ["x.md", "y.md"]
        relevant = ["a.md", "b.md"]

        # When
        result = recall_at_k(retrieved, relevant, k=2)

        # Then
        assert result == 0.0

    def test_k_limits_search(self) -> None:
        """Given k smaller than retrieved list, then only top k is considered."""
        # Given
        retrieved = ["x.md", "a.md"]
        relevant = ["a.md"]

        # When
        result = recall_at_k(retrieved, relevant, k=1)

        # Then
        assert result == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        """Given empty relevant list, then recall is 0.0."""
        # Given / When
        result = recall_at_k(["a.md"], [], k=1)

        # Then
        assert result == 0.0

    def test_k_zero_returns_zero(self) -> None:
        """Given k=0, then recall is 0.0."""
        # Given / When
        result = recall_at_k(["a.md"], ["a.md"], k=0)

        # Then
        assert result == 0.0

    def test_k_negative_returns_zero(self) -> None:
        """Given negative k, then recall is 0.0."""
        # Given / When
        result = recall_at_k(["a.md"], ["a.md"], k=-1)

        # Then
        assert result == 0.0


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """Given all top k are relevant, then precision is 1.0."""
        # Given
        retrieved = ["a.md", "b.md"]
        relevant = ["a.md", "b.md", "c.md"]

        # When
        result = precision_at_k(retrieved, relevant, k=2)

        # Then
        assert result == 1.0

    def test_partial_precision(self) -> None:
        """Given some top k are relevant, then precision is fractional."""
        # Given
        retrieved = ["a.md", "x.md"]
        relevant = ["a.md"]

        # When
        result = precision_at_k(retrieved, relevant, k=2)

        # Then
        assert result == 0.5

    def test_zero_precision(self) -> None:
        """Given no top k are relevant, then precision is 0.0."""
        # Given
        retrieved = ["x.md", "y.md"]
        relevant = ["a.md"]

        # When
        result = precision_at_k(retrieved, relevant, k=2)

        # Then
        assert result == 0.0

    def test_k_zero_returns_zero(self) -> None:
        """Given k=0, then precision is 0.0."""
        # Given / When
        result = precision_at_k(["a.md"], ["a.md"], k=0)

        # Then
        assert result == 0.0

    def test_k_negative_returns_zero(self) -> None:
        """Given negative k, then precision is 0.0."""
        # Given / When
        result = precision_at_k(["a.md"], ["a.md"], k=-1)

        # Then
        assert result == 0.0


class TestScoredQuery:
    """Tests for ScoredQuery dataclass."""

    def test_creates_with_values(self) -> None:
        """Given all fields, then the dataclass stores them."""
        # Given / When
        sq = ScoredQuery(
            query_id="q1",
            category="DIRECT_LOOKUP",
            recall_at_k=0.8,
            precision_at_k=0.6,
        )

        # Then
        assert sq.query_id == "q1"
        assert sq.category == "DIRECT_LOOKUP"
        assert sq.recall_at_k == 0.8
        assert sq.precision_at_k == 0.6

    def test_is_frozen(self) -> None:
        """Given a ScoredQuery, then it cannot be mutated."""
        # Given
        from dataclasses import FrozenInstanceError

        sq = ScoredQuery(
            query_id="q1", category="c", recall_at_k=1.0, precision_at_k=1.0
        )

        # When / Then
        with pytest.raises(FrozenInstanceError):
            sq.query_id = "q2"  # type: ignore


class TestAggregateScores:
    """Tests for _aggregate_scores internal function."""

    def test_empty_returns_zeros(self) -> None:
        """Given no items, then zero counts are returned."""
        # Given / When
        result = _aggregate_scores([])

        # Then
        assert result == {"recall": 0.0, "precision": 0.0, "count": 0.0}

    def test_single_item(self) -> None:
        """Given one item, then its scores are the mean."""
        # Given
        items = [
            ScoredQuery(
                query_id="q1", category="c", recall_at_k=0.8, precision_at_k=0.6
            )
        ]

        # When
        result = _aggregate_scores(items)

        # Then
        assert result["recall"] == 0.8
        assert result["precision"] == 0.6
        assert result["count"] == 1.0

    def test_multiple_items_averages(self) -> None:
        """Given multiple items, then scores are averaged."""
        # Given
        items = [
            ScoredQuery(
                query_id="q1", category="c", recall_at_k=0.8, precision_at_k=0.6
            ),
            ScoredQuery(
                query_id="q2", category="c", recall_at_k=0.4, precision_at_k=0.2
            ),
        ]

        # When
        result = _aggregate_scores(items)

        # Then
        assert abs(result["recall"] - 0.6) < 1e-9
        assert abs(result["precision"] - 0.4) < 1e-9
        assert result["count"] == 2.0


class TestAggregateByCategory:
    """Tests for aggregate_by_category function."""

    def test_empty_input_returns_overall_only(self) -> None:
        """Given no scores, then only OVERALL entry is returned."""
        # Given / When
        result = aggregate_by_category([])

        # Then
        assert list(result.keys()) == [OVERALL_CATEGORY]
        assert result[OVERALL_CATEGORY]["count"] == 0.0

    def test_single_category(self) -> None:
        """Given scores from one category, then per-category and overall match."""
        # Given
        scores = [
            ScoredQuery(
                query_id="q1", category="A", recall_at_k=0.8, precision_at_k=0.6
            ),
            ScoredQuery(
                query_id="q2", category="A", recall_at_k=0.4, precision_at_k=0.2
            ),
        ]

        # When
        result = aggregate_by_category(scores)

        # Then
        assert "A" in result
        assert abs(result["A"]["recall"] - 0.6) < 1e-9
        assert result["A"]["count"] == 2.0
        assert abs(result[OVERALL_CATEGORY]["recall"] - 0.6) < 1e-9

    def test_multiple_categories(self) -> None:
        """Given scores from multiple categories, then each is aggregated separately."""
        # Given
        scores = [
            ScoredQuery(
                query_id="q1", category="A", recall_at_k=1.0, precision_at_k=1.0
            ),
            ScoredQuery(
                query_id="q2", category="B", recall_at_k=0.0, precision_at_k=0.0
            ),
        ]

        # When
        result = aggregate_by_category(scores)

        # Then
        assert result["A"]["recall"] == 1.0
        assert result["B"]["recall"] == 0.0
        assert result[OVERALL_CATEGORY]["recall"] == 0.5
        assert result[OVERALL_CATEGORY]["count"] == 2.0

    def test_overall_includes_all_categories(self) -> None:
        """Given mixed categories, then OVERALL aggregates everything."""
        # Given
        scores = [
            ScoredQuery(
                query_id="q1", category="A", recall_at_k=1.0, precision_at_k=0.5
            ),
            ScoredQuery(
                query_id="q2", category="B", recall_at_k=0.0, precision_at_k=0.5
            ),
            ScoredQuery(
                query_id="q3", category="A", recall_at_k=0.5, precision_at_k=1.0
            ),
        ]

        # When
        result = aggregate_by_category(scores)

        # Then
        assert result[OVERALL_CATEGORY]["count"] == 3.0
        expected_recall = (1.0 + 0.0 + 0.5) / 3
        assert abs(result[OVERALL_CATEGORY]["recall"] - expected_recall) < 1e-9
