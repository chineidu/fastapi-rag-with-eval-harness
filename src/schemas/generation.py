"""Structured answer and prompt types for Slice 2 generation."""

from dataclasses import dataclass

from pydantic import Field

from src.schemas.base import BaseSchema

__all__ = ["GeneratedAnswer", "GenerationPrompt"]


@dataclass(slots=True, frozen=True)
class GenerationPrompt:
    """Assembled prompt pair for testing without network.

    Parameters
    ----------
    system : str
        System instruction with grounding rules and few-shot examples.
    user : str
        Query plus formatted chunk contexts.

    """

    system: str
    user: str


class GeneratedAnswer(BaseSchema):
    """LLM answer with citations (instructor response model)."""

    answer: str = Field(min_length=1, description="Grounded answer text.")
    citations: list[str] = Field(
        default_factory=list, description="Cited doc_path values from context."
    )
    model_id: str = Field(min_length=1, description="OpenRouter model identifier.")
    grounded: bool = Field(description="False when the model answers unknown.")
