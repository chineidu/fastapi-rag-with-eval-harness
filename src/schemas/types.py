"""Shared enums and named tuples for the eval pipeline."""

from enum import StrEnum
from typing import NamedTuple


# ======= Enums =======
class EnvironmentEnum(StrEnum):
    """Runtime environments the application can run in."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    SANDBOX = "sandbox"


class ErrorCodeEnum(StrEnum):
    """Error codes used in API responses."""

    HTTP_ERROR = "http_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    UNAUTHORIZED = "unauthorized"
    UNEXPECTED_ERROR = "unexpected_error"
    INVALID_INPUT = "invalid_input"
    GENERATION_ERROR = "generation_error"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT_ERROR = "timeout_error"


class ClassificationLabel(StrEnum):
    """Classification labels for the eval dataset."""

    DIRECT_LOOKUP = "DIRECT_LOOKUP"
    MULTI_HOP = "MULTI_HOP"
    CONCEPTUAL = "CONCEPTUAL"
    UNKNOWN = "UNKNOWN"


class EmbeddingProviderEnum(StrEnum):
    """Embedding providers for the eval dataset."""

    LOCAL = "local"
    API = "api"
    # Deterministic fake for tests/offline pipelines; no model weights or network access.
    STUB = "stub"


class VectorStoreBackendEnum(StrEnum):
    """Vector store backends for the document indexer."""

    QDRANT = "qdrant"


class ChunkStrategyEnum(StrEnum):
    """Chunking strategies for the document indexer."""

    NAIVE = "naive"
    STRUCTURAL = "structural"


class VerdictEnum(StrEnum):
    """Binary relevance verdicts for the LLM judge."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class RunStatus(StrEnum):
    """Lifecycle states for an eval run."""

    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"


class QueryResultStatus(StrEnum):
    """Per-query outcome states stored in ``query_results.status``."""

    SUCCESS = "success"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class DiffFlagEnum(StrEnum):
    """Per-category delta flags for run diffs."""

    IMPROVED = "improved"
    REGRESSED = "regressed"


# ======= NamedTuples =======
class RepoHandle(NamedTuple):
    """A pair of immutable strings representing the owner and name of a GitHub repository."""

    owner: str
    name: str
