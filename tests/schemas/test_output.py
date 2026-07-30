from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.output import (
    DiscussionNodeSchema,
    StackOverflowAnswerSchema,
    StackOverflowQuestionSchema,
    UnifiedEvalRecordSchema,
)


class TestDiscussionNodeSchema:
    def test_validates_from_github_payload(
        self, github_node_payload: dict[str, Any]
    ) -> None:
        node = DiscussionNodeSchema.model_validate(github_node_payload)
        assert node.id == "D_kwDOAFa5CM4A2xYQ"
        assert node.number == 42
        assert node.title == "How to use FastAPI dependencies?"
        assert node.upvote_count == 15
        assert node.is_answered is True
        assert node.state_reason == "RESOLVED"
        assert node.answer is not None
        assert node.answer["body"] == "<p>Use <code>Depends()</code></p>"

    def test_answer_can_be_none(self, github_node_payload: dict[str, Any]) -> None:
        payload = {**github_node_payload, "answer": None}
        node = DiscussionNodeSchema.model_validate(payload)
        assert node.answer is None

    def test_closed_at_can_be_none(self, github_node_payload: dict[str, Any]) -> None:
        payload = {**github_node_payload, "closedAt": None}
        node = DiscussionNodeSchema.model_validate(payload)
        assert node.closed_at is None

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            DiscussionNodeSchema.model_validate({"id": "abc"})


class TestStackOverflowAnswerSchema:
    def test_validates_basic_answer(self) -> None:
        payload = {
            "answer_id": 67890,
            "question_id": 12345,
            "score": 30,
            "is_accepted": True,
            "body": "<p>Answer body</p>",
            "creation_date": 1705399200,
        }
        answer = StackOverflowAnswerSchema.model_validate(payload)
        assert answer.answer_id == 67890
        assert answer.score == 30
        assert answer.is_accepted is True
        assert answer.body_markdown is None

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            StackOverflowAnswerSchema.model_validate({"answer_id": 1})


class TestStackOverflowQuestionSchema:
    def test_validates_with_answer(
        self, stackoverflow_question_payload: dict[str, Any]
    ) -> None:
        question = StackOverflowQuestionSchema.model_validate(
            stackoverflow_question_payload
        )
        assert question.question_id == 12345
        assert question.title == "FastAPI middleware not working"
        assert question.tags == ["fastapi", "middleware", "python"]
        assert question.is_answered is True
        assert question.answer is not None
        assert question.answer.answer_id == 67890
        assert question.answer.score == 30

    def test_answer_can_be_none(
        self, stackoverflow_question_payload: dict[str, Any]
    ) -> None:
        payload = {**stackoverflow_question_payload, "answer": None}
        question = StackOverflowQuestionSchema.model_validate(payload)
        assert question.answer is None


class TestUnifiedEvalRecordSchema:
    def test_validates_unified_record(self) -> None:
        payload = {
            "id": "rec-1",
            "source": "github",
            "title": "Test question",
            "url": "https://example.com/1",
            "body": "question body",
            "answerText": "answer body",
            "createdAt": "2024-01-01T00:00:00Z",
            "score": 10,
            "answerScore": 5,
            "tags": ["fastapi"],
        }
        record = UnifiedEvalRecordSchema.model_validate(payload)
        assert record.id == "rec-1"
        assert record.source == "github"
        assert record.score == 10
