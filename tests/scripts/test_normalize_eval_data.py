from src.schemas.output import DiscussionNodeSchema, StackOverflowQuestionSchema


class TestGitHubToUnified:
    def test_converts_with_answer(
        self, normalize_eval_data, github_node_payload
    ) -> None:
        # Given
        mod = normalize_eval_data
        rec = DiscussionNodeSchema.model_validate(github_node_payload)
        # When
        result = mod._github_to_unified(rec)
        # Then
        assert result.id == "D_kwDOAFa5CM4A2xYQ"
        assert result.source == "github"
        assert result.title == "How to use FastAPI dependencies?"
        assert result.url == "https://github.com/fastapi/fastapi/discussions/42"
        assert result.answer_text == "<p>Use <code>Depends()</code></p>"
        assert result.created_at == "2024-01-15T10:30:00Z"
        assert result.score == 15
        assert result.answer_score == 12
        assert result.tags == ["question", "dependencies"]

    def test_converts_without_answer(
        self, normalize_eval_data, github_node_payload
    ) -> None:
        # Given
        mod = normalize_eval_data
        payload = {**github_node_payload, "answer": None}
        rec = DiscussionNodeSchema.model_validate(payload)
        # When
        result = mod._github_to_unified(rec)
        # Then
        assert result.answer_text == ""
        assert result.answer_score == 0


class TestStackOverflowToUnified:
    def test_converts_with_answer(
        self, normalize_eval_data, stackoverflow_question_payload
    ) -> None:
        # Given
        mod = normalize_eval_data
        rec = StackOverflowQuestionSchema.model_validate(stackoverflow_question_payload)
        # When
        result = mod._stackoverflow_to_unified(rec)
        # Then
        assert result.id == "12345"
        assert result.source == "stackoverflow"
        assert result.title == "FastAPI middleware not working"
        assert result.url == "https://stackoverflow.com/q/12345"
        assert result.body == "I have a custom middleware but it doesn't work."
        assert result.answer_text == "Make sure to add middleware before routes."
        assert result.score == 25
        assert result.answer_score == 30
        assert result.tags == ["fastapi", "middleware", "python"]
        assert "T" in result.created_at
        assert result.created_at.endswith("Z")

    def test_converts_with_answer_body_markdown(
        self, normalize_eval_data, stackoverflow_question_payload
    ) -> None:
        # Given
        mod = normalize_eval_data
        rec = StackOverflowQuestionSchema.model_validate(stackoverflow_question_payload)
        # When
        result = mod._stackoverflow_to_unified(rec)
        # Then: body_markdown is used over stripped HTML for body when available
        assert result.body == "I have a custom middleware but it doesn't work."

    def test_converts_without_answer(
        self, normalize_eval_data, stackoverflow_question_payload
    ) -> None:
        # Given
        mod = normalize_eval_data
        payload = {**stackoverflow_question_payload, "answer": None}
        rec = StackOverflowQuestionSchema.model_validate(payload)
        # When
        result = mod._stackoverflow_to_unified(rec)
        # Then
        assert result.answer_text == ""
        assert result.answer_score == 0
