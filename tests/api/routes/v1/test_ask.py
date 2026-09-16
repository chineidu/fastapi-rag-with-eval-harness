"""Tests for the non-streaming ask endpoint."""

import asyncio
from typing import cast

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.core.dependencies import get_retriever
from src.api.core.exceptions import BaseAPIError
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.generation import GeneratedAnswer
from src.schemas.types import ErrorCodeEnum


def _answer() -> GeneratedAnswer:
    """Canned grounded answer for ask tests."""
    return GeneratedAnswer(
        answer="FastAPI is a web framework.",
        citations=["docs/a.md"],
        model_id="test-model",
        grounded=True,
    )


class FakeRetriever:
    """Retriever stub with scripted generation behavior."""

    def __init__(
        self,
        answer: GeneratedAnswer | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        """Serve the answer, raise the error, or stall for delay seconds."""
        self._answer = answer or _answer()
        self._error = error
        self._delay = delay
        self.last_query: str | None = None
        self.last_k: int | None = None

    async def agenerate(self, query: str, k: int = 10) -> GeneratedAnswer:
        """Record inputs, then answer, fail, or stall as scripted."""
        self.last_query = query
        self.last_k = k
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._answer


def _client(retriever: FakeRetriever) -> TestClient:
    """Test client with the retriever dependency overridden."""
    app = create_app(cast(LocalRetriever, retriever))
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def _prefix() -> str:
    """Route prefix from the bundled config."""
    return "/" + app_config.api_config.prefix.strip("/")


class TestAsk:
    def test_returns_generated_answer(self) -> None:
        """Answer the question and forward query plus depth."""
        # Given
        retriever = FakeRetriever()
        # When
        with _client(retriever) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?"}
            )
        # Then
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "x-request-id" in response.headers
        assert response.json() == {
            "answer": "FastAPI is a web framework.",
            "citations": ["docs/a.md"],
            "modelId": "test-model",
            "grounded": True,
        }
        assert retriever.last_query == "What is FastAPI?"
        assert retriever.last_k == 10

    def test_forwards_top_k(self) -> None:
        """Pass the requested retrieval depth to generation."""
        # Given
        retriever = FakeRetriever()
        # When
        with _client(retriever) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?", "top_k": 7}
            )
        # Then
        assert response.status_code == 200
        assert retriever.last_k == 7

    def test_empty_query_is_invalid(self) -> None:
        """Reject an empty question with the invalid-input envelope."""
        # Given / When
        with _client(FakeRetriever()) as client:
            response = client.post(f"{_prefix()}/ask", json={"query": ""})
        # Then
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"

    def test_top_k_over_ceiling_is_invalid(self) -> None:
        """Reject a depth above the ceiling with the invalid-input envelope."""
        # Given / When
        with _client(FakeRetriever()) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?", "top_k": 101}
            )
        # Then
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"

    def test_generation_failure_maps_to_500(self) -> None:
        """Surface a generation failure as a generation-error envelope."""
        # Given / When
        with _client(FakeRetriever(error=RuntimeError("provider down"))) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?"}
            )
        # Then
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "generation_error"
        assert "Generation error: provider down" in body["error"]["message"]

    def test_slow_generation_maps_to_504(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Surface a stalled generation as a timeout envelope."""
        # Given
        monkeypatch.setattr(app_config.api_config, "timeout", 0.05)
        # When
        with _client(FakeRetriever(delay=5.0)) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?"}
            )
        # Then
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout_error"

    def test_base_api_error_propagates_unchanged(self) -> None:
        """Keep the original status when generation raises a typed error."""
        # Given
        typed = BaseAPIError(
            "quota exceeded",
            status_code=429,
            error_code=ErrorCodeEnum.HTTP_ERROR,
        )
        # When
        with _client(FakeRetriever(error=typed)) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?"}
            )
        # Then
        assert response.status_code == 429
        assert response.json()["error"]["code"] == ErrorCodeEnum.HTTP_ERROR.value
