"""Schemas for the ground-truth labeling pipeline."""

from pydantic import Field

from src.schemas.base import BaseSchema
from src.schemas.models import GroundTruthRecord
from src.schemas.types import ClassificationLabel, VerdictEnum

__all__ = [
    "ClassificationResponse",
    "CorpusDocument",
    "GroundTruthRecord",
    "JudgeResponse",
    "LabelVerdict",
    "VerdictEnum",
]


class CorpusDocument(BaseSchema):
    """A single corpus file with its relative path and text content."""

    path: str = Field(min_length=1, description="Relative file path (doc identifier)")
    content: str = Field(min_length=1, description="Raw text content of the file")


class LabelVerdict(BaseSchema):
    """LLM judge output for a single candidate document."""

    verdict: VerdictEnum = Field(description="Binary relevance judgment")
    rationale: str = Field(min_length=1, description="Short explanation of the verdict")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge confidence in the verdict (0.0-1.0)",
    )


class JudgeResponse(BaseSchema):
    """Structured output from the LLM relevance judge (instructor response model)."""

    verdict: VerdictEnum = Field(description="Binary relevance judgment.")
    rationale: str = Field(description="Short explanation of the verdict.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict.",
    )


class ClassificationResponse(BaseSchema):
    """Structured output from the LLM classification (instructor response model)."""

    chain_of_thought: str = Field(
        description="The chain of thought that led to the prediction. Max 2 sentences.",
    )
    label: ClassificationLabel = Field(description="The predicted class label.")
