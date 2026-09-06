"""Tests for the eval_harness.runner module."""

import pytest

from src.eval_harness.adapter import RetrievalResult
from src.eval_harness.runner import (
    EvalRunner,
    QueryOutcome,
    RunSummary,
    is_recoverable,
    validate_result,
)
from src.eval_harness.store import ResultStore
from src.schemas.models import GroundTruthRecord

RECORD = GroundTruthRecord(
    query_id="q1",
    label="DIRECT_LOOKUP",
    query_text="How to use FastAPI?",
    relevant_docs=["docs/quickstart.md"],
)


class TestIsRecoverable:
    """Tests for is_recoverable function."""

    def test_connection_error_is_recoverable(self) -> None:
        """Given a ConnectionError, then it is recoverable."""
        # Given / When / Then
        assert is_recoverable(ConnectionError("timeout")) is True

    def test_timeout_error_is_recoverable(self) -> None:
        """Given a TimeoutError, then it is recoverable."""
        # Given / When / Then
        assert is_recoverable(TimeoutError("timed out")) is True

    def test_value_error_is_not_recoverable(self) -> None:
        """Given a ValueError, then it is not recoverable."""
        # Given / When / Then
        assert is_recoverable(ValueError("bad input")) is False

    def test_http_5xx_status_code_is_recoverable(self) -> None:
        """Given an exception with status_code=500, then it is recoverable."""
        # Given
        exc = Exception()
        exc.status_code = 500  # type: ignore

        # When / Then
        assert is_recoverable(exc) is True

    def test_http_4xx_status_code_is_not_recoverable(self) -> None:
        """Given an exception with status_code=404, then it is not recoverable."""
        # Given
        exc = Exception()
        exc.status_code = 404  # type: ignore

        # When / Then
        assert is_recoverable(exc) is False

    def test_nested_response_5xx_is_recoverable(self) -> None:
        """Given an exception with response.status_code=503, then it is recoverable."""
        # Given
        exc = Exception()
        exc.response = Exception()  # type: ignore
        exc.response.status_code = 503  # type: ignore

        # When / Then
        assert is_recoverable(exc) is True

    def test_non_int_status_code_is_not_recoverable(self) -> None:
        """Given an exception with non-int status_code, then it is not recoverable."""
        # Given
        exc = Exception()
        exc.status_code = "500"  # type: ignore

        # When / Then
        assert is_recoverable(exc) is False


class TestValidateResult:
    """Tests for validate_result function."""

    def test_valid_result_passes(self) -> None:
        """Given a valid RetrievalResult, then it is returned unchanged."""
        # Given
        result = RetrievalResult(documents=[("a.md", 0.9)])

        # When
        validated = validate_result(result)

        # Then
        assert validated is result

    def test_non_retrieval_result_raises_type_error(self) -> None:
        """Given a non-RetrievalResult, then TypeError is raised."""
        # Given / When / Then
        with pytest.raises(TypeError, match="Adapter must return RetrievalResult"):
            validate_result("not a result")

    def test_wrong_entry_length_raises(self) -> None:
        """Given a document entry with wrong length, then ValueError is raised."""
        # Given
        result = RetrievalResult(documents=[("a.md", 0.9, "extra")])  # type: ignore

        # When / Then
        with pytest.raises(ValueError, match="Malformed"):
            validate_result(result)

    def test_non_string_doc_path_raises(self) -> None:
        """Given a document entry with non-string path, then ValueError is raised."""
        # Given
        result = RetrievalResult(documents=[(123, 0.9)])  # type: ignore

        # When / Then
        with pytest.raises(ValueError, match="Malformed"):
            validate_result(result)

    def test_non_numeric_score_raises(self) -> None:
        """Given a document entry with non-numeric score, then ValueError is raised."""
        # Given
        result = RetrievalResult(documents=[("a.md", "high")])  # type: ignore

        # When / Then
        with pytest.raises(ValueError, match="Malformed"):
            validate_result(result)

    def test_list_entry_is_accepted(self) -> None:
        """Given a document entry as list, then it is accepted."""
        # Given
        result = RetrievalResult(documents=[["a.md", 0.9]])  # type: ignore

        # When
        validated = validate_result(result)

        # Then
        assert validated.documents[0] == ["a.md", 0.9]


class TestQueryOutcome:
    """Tests for QueryOutcome dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Given required fields, then the dataclass stores them."""
        # Given / When
        outcome = QueryOutcome(
            query_id="q1",
            category="A",
            retrieved_docs=[],
            recall_at_k=0.8,
            precision_at_k=0.6,
            latency_ms=12.5,
            status="success",
        )

        # Then
        assert outcome.query_id == "q1"
        assert outcome.error is None

    def test_error_defaults_to_none(self) -> None:
        """Given no error, then error defaults to None."""
        # Given / When
        outcome = QueryOutcome(
            query_id="q1",
            category="A",
            retrieved_docs=[],
            recall_at_k=None,
            precision_at_k=None,
            latency_ms=0.0,
            status="fatal",
        )

        # Then
        assert outcome.error is None


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_creates_with_required_fields(self) -> None:
        """Given required fields, then the dataclass stores them."""
        # Given / When
        summary = RunSummary(
            run_id=1,
            tag="test-run",
            status="complete",
            total=10,
            completed=10,
            failed=0,
        )

        # Then
        assert summary.run_id == 1
        assert summary.tag == "test-run"
        assert summary.category_scores == {}

    def test_category_scores_defaults_to_empty(self) -> None:
        """Given no category_scores, then it defaults to empty dict."""
        # Given / When
        summary = RunSummary(
            run_id=1, tag="t", status="c", total=1, completed=1, failed=0
        )

        # Then
        assert summary.category_scores == {}


class TestEvalRunner:
    """Tests for EvalRunner class."""

    def test_rejects_empty_ground_truth(self, tmp_path) -> None:
        """Given empty ground_truth, then ValueError is raised."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()

        # When / Then
        with pytest.raises(ValueError, match="ground_truth must not be empty"):
            EvalRunner(adapter, store, [])

    def test_rejects_zero_k(self, tmp_path) -> None:
        """Given k=0, then ValueError is raised."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()

        # When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            EvalRunner(adapter, store, [RECORD], k=0)

    def test_rejects_negative_concurrency(self, tmp_path) -> None:
        """Given negative concurrency, then ValueError is raised."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()

        # When / Then
        with pytest.raises(ValueError, match="concurrency must be positive"):
            EvalRunner(adapter, store, [RECORD], concurrency=-1)

    def test_run_returns_summary(self, tmp_path) -> None:
        """Given a valid runner, then run() returns a RunSummary."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, [RECORD], k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("test-run")

        # Then
        assert isinstance(summary, RunSummary)
        assert summary.tag == "test-run"
        assert summary.total == 1
        assert summary.completed == 1
        assert summary.failed == 0
        assert summary.status == "complete"

    def test_run_persists_results(self, tmp_path) -> None:
        """Given a run, then results are persisted in the store."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, [RECORD], k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("test-run")

        # Then
        results = store.get_results(summary.run_id)
        assert len(results) == 1
        assert results[0]["query_id"] == "q1"
        assert results[0]["status"] == "success"

    def test_run_with_fatal_error(self, tmp_path) -> None:
        """Given an adapter that raises a non-recoverable error, then status is fatal."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = FailingAdapter(ValueError("bad input"))
        runner = EvalRunner(adapter, store, [RECORD], k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("test-run")

        # Then
        assert summary.failed == 1
        assert summary.status == "partial"

    def test_run_with_recoverable_error_exhausted(self, tmp_path) -> None:
        """Given a ConnectionError that exhausts retries, then status is recoverable."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = FailingAdapter(ConnectionError("network"))
        runner = EvalRunner(
            adapter,
            store,
            [RECORD],
            k=1,
            concurrency=1,
            max_retries=2,
            backoff_base_secs=0.0,
        )

        # When
        summary = runner.run("test-run")

        # Then
        assert summary.failed == 1
        assert summary.status == "partial"

    def test_run_with_config_and_metadata(self, tmp_path) -> None:
        """Given config and metadata, then they are stored in the run row."""
        # Given
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, [RECORD], k=1, concurrency=1, max_retries=0)
        config = {"adapter": "stub"}
        metadata = {"note": "testing"}

        # When
        summary = runner.run("test-run", config=config, metadata=metadata)

        # Then
        run_row = store.get_run(summary.run_id)
        assert run_row is not None
        assert run_row["tag"] == "test-run"
        assert run_row["config"] is not None
        assert run_row["metadata"] is not None

    def test_multiple_queries(self, tmp_path) -> None:
        """Given multiple ground truth records, then all are evaluated."""
        # Given
        records = [
            GroundTruthRecord(
                query_id=f"q{i}",
                label="DIRECT_LOOKUP",
                query_text=f"Question {i}",
                relevant_docs=[f"doc{i}.md"],
            )
            for i in range(3)
        ]
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, records, k=1, concurrency=2, max_retries=0)

        # When
        summary = runner.run("multi-run")

        # Then
        assert summary.total == 3
        assert summary.completed == 3
        results = store.get_results(summary.run_id)
        assert len(results) == 3

    def test_unanswerable_excluded_from_scores(self, tmp_path) -> None:
        """Given unanswerable records, then they are counted but excluded from means."""
        # Given
        records = [
            GroundTruthRecord(
                query_id="q1",
                label="DIRECT_LOOKUP",
                query_text="Question 1",
                relevant_docs=["doc1.md"],
            ),
            GroundTruthRecord(
                query_id="q2",
                label="MULTI_HOP",
                query_text="Question 2",
                relevant_docs=[],
                answerable=False,
            ),
        ]
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, records, k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("unanswerable-run")

        # Then
        assert summary.total == 2
        assert summary.completed == 2
        assert summary.unanswerable == 1
        # Only the answerable query contributes to the OVERALL mean.
        overall = summary.category_scores["OVERALL"]
        assert overall["count"] == 1.0

    def test_all_unanswerable_returns_zero_scores(self, tmp_path) -> None:
        """Given only unanswerable records, then OVERALL scores are zero with count 0."""
        # Given
        records = [
            GroundTruthRecord(
                query_id=f"q{i}",
                label="DIRECT_LOOKUP",
                query_text=f"Question {i}",
                relevant_docs=[],
                answerable=False,
            )
            for i in range(2)
        ]
        store = ResultStore(tmp_path / "test.db")
        adapter = StubAdapter()
        runner = EvalRunner(adapter, store, records, k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("all-unanswerable")

        # Then
        assert summary.total == 2
        assert summary.completed == 2
        assert summary.unanswerable == 2
        overall = summary.category_scores["OVERALL"]
        assert overall["count"] == 0.0
        assert overall["recall"] == 0.0
        assert overall["precision"] == 0.0

    def test_unanswerable_failure_counts_both_failed_and_unanswerable(
        self, tmp_path
    ) -> None:
        """Given an unanswerable record that fails, then it counts in both buckets."""
        # Given
        record = GroundTruthRecord(
            query_id="q1",
            label="MULTI_HOP",
            query_text="Question 1",
            relevant_docs=[],
            answerable=False,
        )
        store = ResultStore(tmp_path / "test.db")
        adapter = FailingAdapter(ValueError("bad input"))
        runner = EvalRunner(adapter, store, [record], k=1, concurrency=1, max_retries=0)

        # When
        summary = runner.run("fail-unanswerable")

        # Then
        assert summary.failed == 1
        assert summary.unanswerable == 1
        assert summary.completed == 0


class StubAdapter:
    """Minimal adapter that returns deterministic results for testing."""

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        return RetrievalResult(documents=[("docs/quickstart.md", 0.9)])

    def generate(self, query: str, documents: list[str]) -> str:
        return "stub answer"


class FailingAdapter:
    """Adapter that raises a configurable exception on retrieve."""

    def __init__(self, exc: Exception) -> None:
        """Initialize with the exception to raise.

        Parameters
        ----------
        exc : Exception
            The exception to raise on retrieve().

        """
        self._exc = exc

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        raise self._exc

    def generate(self, query: str, documents: list[str]) -> str:
        raise self._exc
