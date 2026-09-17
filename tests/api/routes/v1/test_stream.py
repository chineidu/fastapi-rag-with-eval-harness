"""Tests for the streaming ask endpoint."""

import asyncio
import json
from typing import cast

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.core.dependencies import get_retriever
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.generation import GeneratedAnswer


class FakeRetriever:
    """Retriever stub with scripted stream behavior."""

    def __init__(
        self,
        partials: list[GeneratedAnswer] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
        fail_at: int | None = None,
    ) -> None:
        """Yield the partials, stall, or fail immediately or at an index."""
        self._partials = partials or [
            GeneratedAnswer(answer="Fast", citations=[], model_id="m", grounded=True),
            GeneratedAnswer(
                answer="FastAPI",
                citations=["docs/a.md"],
                model_id="m",
                grounded=True,
            ),
        ]
        self._error = error
        self._delay = delay
        self._fail_at = fail_at

    async def astream(self, query: str, k: int = 10):
        """Stall, fail, or yield the scripted partials."""
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None and self._fail_at is None:
            raise self._error
        for index, partial in enumerate(self._partials):
            if self._error is not None and index == self._fail_at:
                raise self._error
            yield partial


def _client(retriever: FakeRetriever) -> TestClient:
    """Test client with the retriever dependency overridden."""
    app = create_app(cast(LocalRetriever, retriever))
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def _prefix() -> str:
    """Route prefix from the bundled config."""
    return "/" + app_config.api_config.prefix.strip("/")


def _frames(text: str) -> list[str]:
    """Split a server-sent-event body into data payloads."""
    return [
        block.removeprefix("data: ").strip()
        for block in text.strip().split("\n\n")
        if block
    ]


class TestAskStream:
    def test_streams_snapshots_then_done(self) -> None:
        """Emit one event per snapshot plus a terminal done marker."""
        # Given / When
        with _client(FakeRetriever()) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _frames(response.text)
        assert frames[-1] == "[DONE]"
        assert len(frames) == 3
        assert json.loads(frames[0])["answer"] == "Fast"
        last = json.loads(frames[-2])
        assert last["answer"] == "FastAPI"
        assert last["modelId"] == "m"

    def test_empty_query_is_invalid(self) -> None:
        """Reject an empty question before streaming starts."""
        # Given / When
        with _client(FakeRetriever()) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": ""})
        # Then
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"

    def test_prefirstbyte_failure_maps_to_500(self) -> None:
        """Surface a pre-first-byte failure as a generation envelope."""
        # Given / When
        with _client(FakeRetriever(error=RuntimeError("provider down"))) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "generation_error"

    def test_stalled_first_token_maps_to_504(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surface a stalled first token as a timeout envelope."""
        # Given
        monkeypatch.setattr(app_config.api_config, "timeout", 0.05)
        # When
        with _client(FakeRetriever(delay=5.0)) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout_error"

    def test_midstream_failure_truncates_without_done(self) -> None:
        """A mid-stream failure ends the body without the done marker."""
        # Given / When
        with _client(FakeRetriever(error=RuntimeError("cut"), fail_at=1)) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then
        assert response.status_code == 200
        frames = _frames(response.text)
        assert len(frames) == 1
        assert "[DONE]" not in frames
