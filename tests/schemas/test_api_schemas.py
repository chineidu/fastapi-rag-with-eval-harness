import pytest
from pydantic import ValidationError

from src.schemas.api import AskRequest, HealthResponse, ReadinessResponse
from src.schemas.generation import GeneratedAnswer


class TestAskRequest:
    def test_defaults(self) -> None:
        # Given
        data = {"query": "What is FastAPI?"}
        # When
        request = AskRequest(**data)
        # Then
        assert request.query == "What is FastAPI?"
        assert request.top_k == 10

    def test_rejects_empty_query(self) -> None:
        """A whitespace-only query is stripped to empty and rejected."""
        # Given
        data = {"query": "   "}
        # When / Then
        with pytest.raises(ValidationError):
            AskRequest(**data)

    def test_rejects_out_of_range_top_k(self) -> None:
        # Given
        data = {"query": "What is FastAPI?"}
        # When / Then
        with pytest.raises(ValidationError):
            AskRequest(**data, top_k=0)
        with pytest.raises(ValidationError):
            AskRequest(**data, top_k=101)

    def test_camel_case_alias(self) -> None:
        # Given
        data = {"query": "What is FastAPI?", "topK": 5}
        # When
        request = AskRequest(**data)
        # Then
        assert request.top_k == 5
        assert request.model_dump(by_alias=True)["topK"] == 5


class TestHealthResponse:
    def test_fields(self) -> None:
        # Given
        data = {"name": "api", "status": "healthy", "version": "v1.0.0"}
        # When
        response = HealthResponse(**data)
        # Then
        assert response.name == "api"
        assert response.status == "healthy"
        assert response.version == "v1.0.0"

    def test_rejects_empty_fields(self) -> None:
        # Given
        data = {"name": "", "status": "healthy", "version": "v1.0.0"}
        # When / Then
        with pytest.raises(ValidationError):
            HealthResponse(**data)


class TestReadinessResponse:
    def test_defaults_to_not_ready(self) -> None:
        # Given / When
        response = ReadinessResponse(ready=False)
        # Then
        assert response.ready is False
        assert response.collection is None
        assert response.chunk_count == 0

    def test_ready_with_collection(self) -> None:
        # Given / When
        response = ReadinessResponse(
            ready=True, collection="fastapi_docs", chunk_count=42
        )
        # Then
        assert response.ready is True
        assert response.collection == "fastapi_docs"
        assert response.chunk_count == 42


class TestGeneratedAnswerAlias:
    def test_model_id_renders_camel_case(self) -> None:
        # Given
        answer = GeneratedAnswer(
            answer="FastAPI is a framework.",
            citations=["docs/index.md"],
            model_id="deepseek/deepseek-v4-flash",
            grounded=True,
        )
        # When
        dumped = answer.model_dump(by_alias=True)
        # Then
        assert dumped["modelId"] == "deepseek/deepseek-v4-flash"
