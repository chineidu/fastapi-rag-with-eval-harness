from enum import StrEnum
from typing import NamedTuple


# ======= Enums =======
class EnvironmentEnum(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    SANDBOX = "sandbox"


class ErrorCodeEnum(StrEnum):
    HTTP_ERROR = "http_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    UNAUTHORIZED = "unauthorized"
    UNEXPECTED_ERROR = "unexpected_error"


class ClassificationLabel(StrEnum):
    """Classification labels for the eval dataset."""

    DIRECT_LOOKUP = "DIRECT_LOOKUP"
    MULTI_HOP = "MULTI_HOP"
    CONCEPTUAL = "CONCEPTUAL"


class EmbeddingProviderEnum(StrEnum):
    """Embedding providers for the eval dataset."""

    LOCAL = "local"
    API = "api"
    # Deterministic fake for tests/offline pipelines; no model weights or network access.
    STUB = "stub"


# ======= NamedTuples =======
class RepoHandle(NamedTuple):
    """A pair of immutable strings representing the owner and name of a GitHub repository."""

    owner: str
    name: str
