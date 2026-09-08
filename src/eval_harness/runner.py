"""EvalRunner: orchestrates queries, concurrency, timing, and error handling."""

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures._base import Future
from typing import Any

from src import create_logger
from src.eval_harness.adapter import RetrieverAdapter
from src.eval_harness.metrics import (
    aggregate_by_category,
    precision_at_k,
    recall_at_k,
)
from src.eval_harness.store import QueryResultStatus, ResultStore, RunStatus
from src.schemas.harness import (
    QueryOutcome,
    RetrievalResult,
    RetrievedDocument,
    RunSummary,
    ScoredQuery,
)
from src.schemas.models import GroundTruthRecord

logger = create_logger(name=__name__)

RECOVERABLE_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


def is_recoverable(exc: BaseException) -> bool:
    """Decide whether an adapter exception is worth retrying.

    Parameters
    ----------
    exc : BaseException
        The exception raised by ``adapter.retrieve()``.

    Returns
    -------
    bool
        ``True`` for transient failures (``ConnectionError``,
        ``TimeoutError``, HTTP 5xx carried as ``status_code`` or
        ``response.status_code``); ``False`` otherwise.

    """
    if isinstance(exc, RECOVERABLE_ERRORS):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 500 <= status <= 599:
        return True
    response = getattr(exc, "response", None)
    inner = getattr(response, "status_code", None)
    return isinstance(inner, int) and 500 <= inner <= 599


def validate_result(result: object) -> RetrievalResult:
    """Validate the shape of an adapter result.

    Parameters
    ----------
    result : object
        Value returned by ``adapter.retrieve()``.

    Returns
    -------
    RetrievalResult
        The validated result.

    Raises
    ------
    TypeError
        If ``result`` is not a ``RetrievalResult`` or entries are not
        ``RetrievedDocument``.

    """
    if not isinstance(result, RetrievalResult):
        raise TypeError(
            f"Adapter must return RetrievalResult, got {type(result).__name__}"
        )
    for entry in result.documents:
        if not isinstance(entry, RetrievedDocument):
            raise TypeError(f"Malformed retrieved document entry: {entry!r}")
    return result


class EvalRunner:
    """Run an adapter over ground truth queries with bounded concurrency."""

    def __init__(
        self,
        adapter: RetrieverAdapter,
        store: ResultStore,
        ground_truth: Sequence[GroundTruthRecord],
        *,
        k: int = 10,
        concurrency: int = 3,
        max_retries: int = 2,
        backoff_base_secs: float = 0.5,
    ) -> None:
        """Configure the runner.

        Parameters
        ----------
        adapter : RetrieverAdapter
            Project-specific retriever under evaluation.
        store : ResultStore
            Persistence for runs and per-query results.
        ground_truth : Sequence[GroundTruthRecord]
            Queries to evaluate. Must be non-empty.
        k : int
            Retrieval depth passed to ``adapter.retrieve()``.
        concurrency : int
            Max parallel adapter calls.
        max_retries : int
            Retries for recoverable errors (total attempts = 1 + max_retries).
        backoff_base_secs : float
            Base for exponential backoff between retries.

        Raises
        ------
        ValueError
            If ``ground_truth`` is empty, or ``k``/``concurrency`` is
            not positive.

        """
        if len(ground_truth) == 0:
            raise ValueError("ground_truth must not be empty")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        self._adapter = adapter
        self._store = store
        self._ground_truth = list(ground_truth)
        self._k = k
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._backoff_base_secs = backoff_base_secs

    def run(
        self,
        tag: str,
        *,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunSummary:
        """Evaluate every ground truth query and persist the results.

        Parameters
        ----------
        tag : str
            Non-unique label for this run.
        config : Mapping[str, Any] | None
            Run config snapshot (k, concurrency, adapter, ...) for the run row.
        metadata : Mapping[str, Any] | None
            Freeform notes for the run row.

        Returns
        -------
        RunSummary
            Totals plus per-category mean scores.

        """
        run_config: dict[str, Any | int] = {
            "k": self._k,
            "concurrency": self._concurrency,
        }
        if config is not None:
            run_config.update(dict(config))
        run_id: int = self._store.create_run(tag, config=run_config, metadata=metadata)

        outcomes: list[QueryOutcome] = []
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            futures: dict[Future[QueryOutcome], str] = {
                pool.submit(self._run_single, record, run_id): record.query_id
                for record in self._ground_truth
            }
            outcomes.extend(future.result() for future in as_completed(futures))

        successful = [
            outcome
            for outcome in outcomes
            if outcome.status == QueryResultStatus.SUCCESS.value
        ]
        # recall@k means are computed over answerable queries only; unanswerable
        # queries are counted separately so they do not drag the mean to zero.
        scored = [
            ScoredQuery(
                query_id=outcome.query_id,
                category=outcome.category,
                recall_at_k=outcome.recall_at_k or 0.0,
                precision_at_k=outcome.precision_at_k or 0.0,
            )
            for outcome in successful
            if outcome.answerable
        ]
        category_scores = aggregate_by_category(scored)

        unanswerable = sum(1 for record in self._ground_truth if not record.answerable)
        completed = len(successful)
        failed = len(outcomes) - completed
        status = RunStatus.COMPLETE.value if failed == 0 else RunStatus.PARTIAL.value
        # Mark as completed
        self._store.finalize_run(run_id, status)

        logger.info(
            "Run %d complete: %d/%d succeeded (%d unanswerable)",
            run_id,
            completed,
            len(outcomes),
            unanswerable,
        )
        return RunSummary(
            run_id=run_id,
            tag=tag,
            status=status,
            total=len(outcomes),
            completed=completed,
            failed=failed,
            unanswerable=unanswerable,
            category_scores=category_scores,
        )

    def _run_single(self, record: GroundTruthRecord, run_id: int) -> QueryOutcome:
        """Evaluate one query with retry, timing, and persistence.

        Parameters
        ----------
        record : GroundTruthRecord
            Query under evaluation.
        run_id : int
            Owning run id.

        Returns
        -------
        QueryOutcome
            The per-query outcome (never raises for adapter errors).

        """
        latency_ms = 0.0
        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                raw = self._adapter.retrieve(record.query_text, self._k)
                result = validate_result(raw)
            except Exception as exc:  # noqa: BLE001 - adapter errors are recorded, not raised
                latency_ms = (time.monotonic() - start) * 1000.0
                if is_recoverable(exc) and attempt < self._max_retries:
                    backoff = self._backoff_base_secs * (2**attempt)
                    logger.warning(
                        "Recoverable error on %s (attempt %d): %s; retrying in %.2fs",
                        record.query_id,
                        attempt + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                status = (
                    QueryResultStatus.RECOVERABLE.value
                    if is_recoverable(exc)
                    else QueryResultStatus.FATAL.value
                )
                logger.warning("Query %s failed (%s): %s", record.query_id, status, exc)
                self._store.save_result(
                    run_id,
                    record.query_id,
                    record.label,
                    self._k,
                    [],
                    record.relevant_docs,
                    None,
                    None,
                    latency_ms,
                    status=status,
                    error=str(exc),
                )
                return QueryOutcome(
                    query_id=record.query_id,
                    category=record.label,
                    retrieved_docs=[],
                    recall_at_k=None,
                    precision_at_k=None,
                    latency_ms=latency_ms,
                    status=status,
                    error=str(exc),
                    answerable=record.answerable,
                )
            latency_ms = (time.monotonic() - start) * 1000.0
            retrieved_paths = [doc.doc_path for doc in result.documents[: self._k]]
            recall = recall_at_k(retrieved_paths, record.relevant_docs, self._k)
            precision = precision_at_k(retrieved_paths, record.relevant_docs, self._k)
            self._store.save_result(
                run_id,
                record.query_id,
                record.label,
                self._k,
                list(result.documents[: self._k]),
                record.relevant_docs,
                recall,
                precision,
                latency_ms,
                status=QueryResultStatus.SUCCESS.value,
            )
            return QueryOutcome(
                query_id=record.query_id,
                category=record.label,
                retrieved_docs=list(result.documents),
                recall_at_k=recall,
                precision_at_k=precision,
                latency_ms=latency_ms,
                status=QueryResultStatus.SUCCESS.value,
                answerable=record.answerable,
            )
        raise RuntimeError("Unreachable: retry loop exhausted without returning")
