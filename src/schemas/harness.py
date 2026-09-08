"""Harness runtime dataclasses: adapter results, scores, and run outcomes."""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class RetrievedDocument:
    """Single retrieved document with its relevance score.

    Parameters
    ----------
    doc_path : str
        Document identifier (file path).
    score : float
        Relevance score; ``int`` values are coerced to ``float``.

    """

    doc_path: str
    score: float

    def __post_init__(self) -> None:
        """Validate fields and coerce integer scores to float."""
        if not isinstance(self.doc_path, str):
            raise TypeError(f"doc_path must be str, got {type(self.doc_path).__name__}")
        # Reject bool first since bool subclasses int and would coerce to 1.0/0.0.
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be numeric, got {type(self.score).__name__}")
        if isinstance(self.score, int):
            object.__setattr__(self, "score", float(self.score))


@dataclass(slots=True)
class RetrievalResult:
    """Documents retrieved for one query, with optional audit metadata.

    The harness operates at the document level (file paths). Chunk-to-doc
    deduplication is the adapter's responsibility; chunk metadata goes in
    ``metadata`` for auditability.
    """

    documents: list[RetrievedDocument]
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
    retrieved_docs: list[RetrievedDocument]
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
