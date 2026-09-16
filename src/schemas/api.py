"""Request and probe schemas for the Slice 3a API service.

The ask endpoint has no dedicated response schema: it returns
``GeneratedAnswer`` directly so citations travel with the answer.
"""

from pydantic import Field

from src.schemas.base import BaseSchema

__all__ = ["AskRequest", "HealthResponse", "ReadinessResponse"]

MAX_TOP_K = 100


class AskRequest(BaseSchema):
    """Question plus retrieval depth for the non-streaming ask endpoint."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)


class HealthResponse(BaseSchema):
    """Liveness payload; static values from api_config, no I/O."""

    name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ReadinessResponse(BaseSchema):
    """Readiness payload; served with 503 when not ready."""

    ready: bool
    collection: str | None = None
    chunk_count: int = 0
