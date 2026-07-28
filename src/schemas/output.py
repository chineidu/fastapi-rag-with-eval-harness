from typing import Any

from src.schemas.base import BaseSchema


class GitHubAPIResponseSchema(BaseSchema):
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
