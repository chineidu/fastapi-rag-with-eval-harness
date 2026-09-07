"""Harness runtime dataclasses: adapter results, scores, and run outcomes."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalResult:
    """Documents retrieved for one query, with optional audit metadata.

    The harness operates at the document level (file paths). Chunk-to-doc
    deduplication is the adapter's responsibility; chunk metadata goes in
    ``metadata`` for auditability.
    """

    documents: list[tuple[str, float]]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ScoredQuery:
    """Per-query metric values used as input to aggregation."""

    query_id: str
    category: str
    recall_at_k: float
    precision_at_k: float


@dataclass(slots=True)
class QueryOutcome:
    """Per-query outcome of a single eval run."""

    query_id: str
    category: str
    retrieved_docs: list[tuple[str, float]]
    recall_at_k: float | None
    precision_at_k: float | None
    latency_ms: float
    status: str
    error: str | None = None
    answerable: bool = True


@dataclass(slots=True)
class RunSummary:
    """Aggregate outcome of :meth:`EvalRunner.run`."""

    run_id: int
    tag: str
    status: str
    total: int
    completed: int
    failed: int
    unanswerable: int = 0
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
