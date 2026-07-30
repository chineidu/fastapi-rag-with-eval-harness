from typing import Any

from src.schemas.base import BaseSchema
from src.schemas.types import ClassificationLabel


class DiscussionNodeSchema(BaseSchema):
    """Typed representation of a GitHub Discussion API response node."""

    id: str
    number: int
    title: str
    url: str
    body: str
    body_text: str
    created_at: str
    closed_at: str | None
    answer_chosen_at: str | None
    upvote_count: int
    is_answered: bool
    state_reason: str | None
    category: dict[str, str]
    labels: dict[str, Any]
    comments: dict[str, int]
    answer: dict[str, Any] | None


class StackOverflowAnswerSchema(BaseSchema):
    """Typed representation of a Stack Exchange answer."""

    answer_id: int
    question_id: int
    score: int
    is_accepted: bool
    body: str
    body_markdown: str | None = None
    creation_date: int
    link: str | None = None


class StackOverflowQuestionSchema(BaseSchema):
    """Typed representation of a Stack Exchange question paired with its best answer."""

    question_id: int
    title: str
    link: str
    body: str
    body_markdown: str
    score: int
    answer_count: int
    view_count: int
    creation_date: int
    tags: list[str]
    is_answered: bool
    answer: StackOverflowAnswerSchema | None


class UnifiedEvalRecordSchema(BaseSchema):
    """A normalized QA record from GitHub or Stack Overflow for RAG evaluation."""

    id: str
    source: str
    title: str
    url: str
    body: str
    answer_text: str
    created_at: str
    score: int
    answer_score: int
    tags: list[str]
    label: ClassificationLabel | None = None
