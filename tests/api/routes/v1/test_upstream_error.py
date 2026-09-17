"""Tests for provider-unreachable mapping to 503."""

import asyncio
from typing import Any, cast

from fastapi import status
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.core.dependencies import get_retriever
from src.api.core.exceptions import (
    GenerationError,
    UpstreamUnavailableError,
    is_upstream_connection_error,
    map_provider_error,
)
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.generation import GeneratedAnswer
from src.schemas.types import ErrorCodeEnum


class APIConnectionError(Exception):
    """Stub matching the provider SDK connection error name."""


class InstructorRetryException(Exception):  # noqa: N818 - stub must match upstream SDK name
    """Stub matching the instructor retry error name."""


class ConnectError(Exception):
    """Stub matching the httpx transport error name."""


def _answer() -> GeneratedAnswer:
    """Canned answer for upstream tests."""
    return GeneratedAnswer(
        answer="FastAPI is a web framework.",
        citations=["docs/a.md"],
        model_id="test-model",
        grounded=True,
    )


class FakeAskRetriever:
    """Retriever stub failing generation with a scripted error."""

    def __init__(self, error: Exception) -> None:
        """Store the error to raise."""
        self._error = error

    async def agenerate(self, query: str, k: int = 10) -> GeneratedAnswer:
        """Raise the scripted error."""
        raise self._error


class FakeStreamRetriever:
    """Retriever stub failing streaming with a scripted error."""

    def __init__(self, error: Exception) -> None:
        """Store the error to raise."""
        self._error = error

    async def astream(self, query: str, k: int = 10) -> Any:
        """Raise the scripted error before yielding."""
        raise self._error
        yield _answer()  # pragma: no cover - unreachable, keeps generator shape


def _client(retriever: Any) -> TestClient:
    """Test client with the retriever dependency overridden."""
    app = create_app(cast(LocalRetriever, retriever))
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def _prefix() -> str:
    """Route prefix from the bundled config."""
    return "/" + app_config.api_config.prefix.strip("/")


class TestIsUpstreamConnectionError:
    def test_direct_match(self) -> None:
        """Match a provider connection error by class name."""
        # Given
        exc = APIConnectionError("Connection error.")
        # When
        result = is_upstream_connection_error(exc)
        # Then
        assert result is True

    def test_chained_match(self) -> None:
        """Match a wrapped instructor retry error through the cause chain."""
        # Given
        exc = RuntimeError("wrapper")
        exc.__cause__ = InstructorRetryException("Connection error.")
        # When
        result = is_upstream_connection_error(exc)
        # Then
        assert result is True

    def test_dns_message_fallback(self) -> None:
        """Match the errno 8 DNS string without a known class name."""
        # Given
        exc = RuntimeError("[Errno 8] nodename nor servname provided, or not known")
        # When
        result = is_upstream_connection_error(exc)
        # Then
        assert result is True

    def test_generic_error_is_not_upstream(self) -> None:
        """Leave unknown failures for the generic 500 path."""
        # Given
        exc = RuntimeError("provider down")
        # When
        result = is_upstream_connection_error(exc)
        # Then
        assert result is False

    def test_cycle_terminates(self) -> None:
        """Terminate on a cause cycle without hanging."""
        # Given
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first
        # When
        result = is_upstream_connection_error(first)
        # Then
        assert result is False

    def test_explicit_raise_preserves_cause(self) -> None:
        """Preserve the cause chain when mapping with explicit raise."""
        # Given
        original = APIConnectionError("Connection error.")

        class StubLogger:
            def warning(self, msg: str, *args: object) -> None:
                return None

            def exception(self, msg: str, *args: object) -> None:
                return None

        # When
        try:
            raise map_provider_error(
                original,
                logger=cast(Any, StubLogger()),
                operation="Ask",
                query_preview="What?",
            ) from original
        except UpstreamUnavailableError as mapped:
            # Then
            assert mapped.__cause__ is original


class TestUpstreamUnavailableError:
    def test_envelope(self) -> None:
        """Build a 503 error with the upstream code."""
        # Given / When
        exc = UpstreamUnavailableError("LLM provider unreachable, try again later")
        # Then
        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert exc.error_code == ErrorCodeEnum.UPSTREAM_UNAVAILABLE
        assert exc.message.startswith("Upstream unavailable: ")


class TestMapProviderError:
    def test_upstream_uses_caller_logger_warning(self) -> None:
        """Log via the passed logger and return 503 for outages."""
        # Given
        records: list[tuple[str, str]] = []

        class StubLogger:
            def warning(self, msg: str, *args: object) -> None:
                records.append(("warning", msg % args))

            def exception(self, msg: str, *args: object) -> None:
                records.append(("exception", msg % args))

        # When
        mapped = map_provider_error(
            APIConnectionError("Connection error."),
            logger=cast(Any, StubLogger()),
            operation="Ask",
            query_preview="What?",
        )
        # Then
        assert isinstance(mapped, UpstreamUnavailableError)
        assert records and records[0][0] == "warning"

    def test_generic_uses_caller_logger_exception(self) -> None:
        """Log via the passed logger and return 500 for unknowns."""
        # Given
        records: list[tuple[str, str]] = []

        class StubLogger:
            def warning(self, msg: str, *args: object) -> None:
                records.append(("warning", msg % args))

            def exception(self, msg: str, *args: object) -> None:
                records.append(("exception", msg % args))

        # When
        mapped = map_provider_error(
            RuntimeError("boom"),
            logger=cast(Any, StubLogger()),
            operation="Stream",
            query_preview="What?",
        )
        # Then
        assert isinstance(mapped, GenerationError)
        assert records and records[0][0] == "exception"


class TestAskUpstream:
    def test_connection_failure_maps_to_503(self) -> None:
        """Surface a provider outage as a sanitized 503 envelope."""
        # Given / When
        with _client(
            FakeAskRetriever(APIConnectionError("Connection error."))
        ) as client:
            response = client.post(
                f"{_prefix()}/ask", json={"query": "What is FastAPI?"}
            )
        # Then
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "upstream_unavailable"
        assert "LLM provider unreachable" in body["error"]["message"]
        assert "Connection error." not in body["error"]["message"]

    def test_slow_generation_still_maps_to_504(self, monkeypatch) -> None:
        """Keep timeout mapping ahead of the upstream check."""
        # Given
        monkeypatch.setattr(app_config.api_config, "timeout", 0.05)

        class SlowRetriever:
            async def agenerate(self, query: str, k: int = 10) -> GeneratedAnswer:
                await asyncio.sleep(5.0)
                return _answer()

        # When
        with _client(SlowRetriever()) as client:
            response = client.post(
                f"{_prefix()}/ask",
                json={"query": "What is FastAPI?"},
            )
        # Then
        assert response.status_code == 504


class TestStreamUpstream:
    def test_prefirstbyte_failure_maps_to_503(self) -> None:
        """Surface a pre-first-byte outage as a sanitized 503 envelope."""
        # Given / When
        with _client(
            FakeStreamRetriever(ConnectError("[Errno 8] nodename nor servname"))
        ) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "upstream_unavailable"
        assert "LLM provider unreachable" in body["error"]["message"]
