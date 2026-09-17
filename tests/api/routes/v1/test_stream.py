"""Tests for the streaming ask endpoint."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.core.dependencies import get_retriever
from src.app.adapter import LocalRetriever
from src.app.generator import RAGGenerator
from src.config import app_config
from src.schemas.generation import GeneratedAnswer
from src.schemas.models import AppConfig
from src.schemas.retrieval import SearchHit


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


def _client(retriever: Any) -> TestClient:
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

    def test_collapse_applies_through_generator(self) -> None:
        """Duplicate and contentless frames collapse end to end."""
        # Given: a leading null frame plus a duplicated partial.
        partial = GeneratedAnswer(
            answer="use", citations=[], model_id="llm-echo", grounded=True
        )
        script: list[Any] = [
            SimpleNamespace(answer=None, citations=[], grounded=None),
            partial,
            partial,
        ]
        aclient = SimpleNamespace(
            chat=SimpleNamespace(completions=_ScriptedCompletions(script))
        )
        generator = RAGGenerator(config=_generator_config(), aclient=aclient)
        contexts = [
            SearchHit(
                chunk_id="docs/a.md#0000",
                doc_path="docs/a.md",
                chunk_index=0,
                text="alpha",
                score=0.9,
            )
        ]
        # When
        with _client(_GeneratorRetriever(generator, contexts)) as client:
            response = client.post(f"{_prefix()}/ask/stream", json={"query": "What?"})
        # Then: one partial event plus the clamped final plus done.
        assert response.status_code == 200
        frames = _frames(response.text)
        assert frames[-1] == "[DONE]"
        assert [json.loads(frame)["answer"] for frame in frames[:-1]] == [
            "use",
            "use",
        ]


class _ScriptedCompletions:
    """Stub chat.completions serving scripted partial frames."""

    def __init__(self, partials: list[Any]) -> None:
        """Store the frames to yield."""
        self._partials = partials

    async def create_partial(self, **kwargs: Any) -> Any:
        """Yield the scripted frames."""
        for partial in self._partials:
            yield partial


class _GeneratorRetriever:
    """Retriever stub delegating streaming to a real RAGGenerator."""

    def __init__(self, generator: RAGGenerator, contexts: list[SearchHit]) -> None:
        """Store the generator and grounding contexts."""
        self._generator = generator
        self._contexts = contexts

    async def astream(self, query: str, k: int = 10) -> Any:
        """Stream collapsed snapshots from the real generator."""
        async for item in self._generator.astream(query, self._contexts):
            yield item


def _generator_config() -> AppConfig:
    """App config stub carrying only the LLM slice the generator reads."""
    return cast(
        AppConfig,
        SimpleNamespace(
            rag_config=SimpleNamespace(
                llm=SimpleNamespace(
                    model_id="test-model",
                    temperature=0.1,
                    max_tokens=256,
                    timeout_seconds=10,
                    max_retries=1,
                    seed=47,
                )
            )
        ),
    )
