"""Copyable retrieval-eval harness: runner, metrics engine, CLI, and SQLite store."""

from src.eval_harness.adapter import RetrievalResult, RetrieverAdapter
from src.eval_harness.config import (
    HarnessConfig,
    HarnessDefaults,
    HarnessDiffThresholds,
    apply_cli_overrides,
    load_harness_config,
)
from src.eval_harness.loader import GroundTruthLoader
from src.eval_harness.metrics import (
    OVERALL_CATEGORY,
    ScoredQuery,
    aggregate_by_category,
    precision_at_k,
    recall_at_k,
)
from src.eval_harness.runner import EvalRunner, QueryOutcome, RunSummary
from src.eval_harness.store import QueryResultStatus, ResultStore, RunStatus
from src.schemas.models import GroundTruthRecord

__all__ = [
    "OVERALL_CATEGORY",
    "EvalRunner",
    "GroundTruthLoader",
    "GroundTruthRecord",
    "HarnessConfig",
    "HarnessDefaults",
    "HarnessDiffThresholds",
    "QueryOutcome",
    "QueryResultStatus",
    "ResultStore",
    "RetrievalResult",
    "RetrieverAdapter",
    "RunStatus",
    "RunSummary",
    "ScoredQuery",
    "aggregate_by_category",
    "apply_cli_overrides",
    "load_harness_config",
    "precision_at_k",
    "recall_at_k",
]
