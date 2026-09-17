"""Tests for RAGGenerator prompt assembly and agenerate (ADR-0027)."""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

import instructor
import pytest

from src.app.generator import RAGGenerator
from src.prompts.generation import ABSTENTION_ANSWER, GENERATE_SYSTEM_PROMPT
from src.schemas.generation import GeneratedAnswer
from src.schemas.models import AppConfig
from src.schemas.retrieval import SearchHit


def _hit(doc_path: str, text: str, score: float = 0.9) -> SearchHit:
    """Build one SearchHit with the given path and text."""
    return SearchHit(
        chunk_id=f"{doc_path}#0000",
        doc_path=doc_path,
        chunk_index=0,
        text=text,
        score=score,
    )


def _config() -> AppConfig:
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


class _FakeCompletions:
    """Stub for aclient.completions with a fixed response or error."""

    def __init__(self, response: GeneratedAnswer | None = None) -> None:
        """Store the response to return."""
        self._response = response
        self.last_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> GeneratedAnswer:
        """Record kwargs and return the fixed response."""
        self.last_kwargs = kwargs
        assert self._response is not None
        return self._response


class _FakeAclient:
    """Stub instructor client exposing .completions.create."""

    def __init__(self, response: GeneratedAnswer | None = None) -> None:
        """Store the response served by completions."""
        self.completions = _FakeCompletions(response)


class TestClientConstruction:
    def test_builds_instructor_client_in_json_schema_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Streaming granularity needs schema mode; JSON mode was flaky."""
        # Given
        captured: dict[str, Any] = {}

        def fake_from_openai(client: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(
            "src.app.generator.instructor.from_openai", fake_from_openai
        )
        # When
        RAGGenerator(config=_config())
        # Then
        assert captured["mode"] is instructor.Mode.JSON_SCHEMA


class TestBuildPrompt:
    def test_system_prompt_carries_abstention_constant(self) -> None:
        """Step 4 and the abstention example cannot drift from the constant."""
        # Given / When
        system = GENERATE_SYSTEM_PROMPT
        # Then
        assert ABSTENTION_ANSWER in system
        assert system.count(ABSTENTION_ANSWER) == 2

    def test_includes_query_and_chunk_paths(self) -> None:
        """The user message carries the query plus labeled chunk blocks."""
        # Given
        generator = RAGGenerator(config=_config(), aclient=_FakeAclient())
        contexts = [_hit("docs/a.md", "alpha"), _hit("docs/b.md", "beta")]
        # When
        prompt = generator.build_prompt("my question", contexts)
        # Then
        assert "<question>\nmy question\n</question>" in prompt.user
        assert "<source>docs/a.md</source>" in prompt.user
        assert "alpha" in prompt.user
        assert "<source>docs/b.md</source>" in prompt.user
        assert prompt.system != ""

    def test_empty_contexts_render_empty_documents(self) -> None:
        """No chunks still yields a well-formed prompt."""
        # Given
        generator = RAGGenerator(config=_config(), aclient=_FakeAclient())
        # When
        prompt = generator.build_prompt("my question", [])
        # Then
        assert "<documents>\n</documents>" in prompt.user


class TestAgenerate:
    async def test_returns_clamped_citations(self) -> None:
        """Unknown citation paths are dropped, order preserved."""
        # Given
        response = GeneratedAnswer(
            answer="use Response",
            citations=["docs/a.md", "docs/ghost.md"],
            model_id="llm-echo",
            grounded=True,
        )
        aclient = _FakeAclient(response)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        result = await generator.agenerate("q", [_hit("docs/a.md", "alpha")])
        # Then
        assert result.citations == ["docs/a.md"]
        assert result.model_id == "test-model"
        assert result.grounded is True
        assert aclient.completions.last_kwargs["model"] == "test-model"

    async def test_preserves_duplicate_citations(self) -> None:
        """Duplicate model citations survive clamping in order."""
        # Given
        response = GeneratedAnswer(
            answer="repeat",
            citations=["docs/a.md", "docs/a.md"],
            model_id="llm-echo",
            grounded=True,
        )
        aclient = _FakeAclient(response)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        result = await generator.agenerate("q", [_hit("docs/a.md", "alpha")])
        # Then
        assert result.citations == ["docs/a.md", "docs/a.md"]

    async def test_preserves_abstention_flag(self) -> None:
        """Model-unknown abstentions keep grounded False with no citations."""
        # Given
        response = GeneratedAnswer(
            answer=ABSTENTION_ANSWER,
            citations=[],
            model_id="llm-echo",
            grounded=False,
        )
        aclient = _FakeAclient(response)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        result = await generator.agenerate("q", [_hit("docs/a.md", "alpha")])
        # Then
        assert result.answer == ABSTENTION_ANSWER
        assert result.grounded is False
        assert result.citations == []

    async def test_reraises_transport_errors(self) -> None:
        """LLM failures propagate instead of masquerading as abstentions."""

        # Given
        class _ExplodingCompletions:
            async def create(self, **kwargs: Any) -> GeneratedAnswer:
                raise ConnectionError("openrouter down")

        aclient = SimpleNamespace(completions=_ExplodingCompletions())
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When / Then
        with pytest.raises(ConnectionError):
            await generator.agenerate("q", [_hit("docs/a.md", "alpha")])


class _FakeStreamingCompletions:
    """Stub for aclient.chat.completions with scripted partials."""

    def __init__(
        self,
        partials: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store the partials to yield or the error to raise."""
        self._partials = partials or []
        self._error = error
        self.last_kwargs: dict[str, Any] = {}

    async def create_partial(self, **kwargs: Any) -> AsyncGenerator[GeneratedAnswer]:
        """Record kwargs, then yield partials or raise on iteration."""
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        for partial in self._partials:
            yield partial


class _FakeStreamingAclient:
    """Stub instructor client exposing chat.completions.create_partial."""

    def __init__(
        self,
        partials: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store the scripted stream behavior."""
        self.chat = SimpleNamespace(
            completions=_FakeStreamingCompletions(partials, error)
        )


class TestAstream:
    async def test_yields_partials_then_clamped_final(self) -> None:
        """Snapshots pass through; only the final yield is clamped."""
        # Given
        partials = [
            GeneratedAnswer(
                answer="use", citations=[], model_id="llm-echo", grounded=True
            ),
            GeneratedAnswer(
                answer="use Response",
                citations=["docs/a.md", "docs/ghost.md"],
                model_id="llm-echo",
                grounded=True,
            ),
        ]
        aclient = _FakeStreamingAclient(partials)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        seen = [
            item async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
        ]
        # Then
        assert len(seen) == 3
        assert seen[0].answer == "use"
        assert seen[1].citations == ["docs/a.md", "docs/ghost.md"]
        assert seen[2].citations == ["docs/a.md"]
        assert seen[2].model_id == "test-model"
        assert aclient.chat.completions.last_kwargs["model"] == "test-model"
        assert aclient.chat.completions.last_kwargs["response_model"] is GeneratedAnswer

    async def test_empty_stream_yields_nothing(self) -> None:
        """A provider that yields nothing produces no events."""
        # Given
        aclient = _FakeStreamingAclient([])
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        seen = [
            item async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
        ]
        # Then
        assert seen == []

    async def test_collapses_noop_snapshots(self) -> None:
        """Provider no-op chunks and the leading null frame never ship."""
        # Given: two contentless frames, then a duplicated partial.
        partial = GeneratedAnswer(
            answer="use", citations=[], model_id="llm-echo", grounded=True
        )
        frames: list[Any] = [
            SimpleNamespace(answer=None, citations=[], grounded=None),
            SimpleNamespace(answer=None, citations=[], grounded=None),
            partial,
            partial,
        ]
        aclient = _FakeStreamingAclient(frames)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        seen = [
            item async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
        ]
        # Then: one partial plus the clamped final, nothing else.
        assert [item.answer for item in seen] == ["use", "use"]
        assert seen[-1].model_id == "test-model"

    async def test_all_contentless_stream_yields_nothing(self) -> None:
        """Frames without content never reach the trailing clamp."""
        # Given: only contentless frames, so the clamp has nothing valid.
        frames: list[Any] = [
            SimpleNamespace(answer=None, citations=[], grounded=None),
            SimpleNamespace(answer="", citations=[], grounded=None),
        ]
        aclient = _FakeStreamingAclient(frames)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        seen = [
            item async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
        ]
        # Then
        assert seen == []

    async def test_trailing_contentless_frame_keeps_final(self) -> None:
        """A contentless tail frame never displaces the clamped final."""
        # Given: valid content followed by a contentless reset frame.
        frames: list[Any] = [
            GeneratedAnswer(
                answer="use", citations=[], model_id="llm-echo", grounded=True
            ),
            SimpleNamespace(answer=None, citations=[], grounded=None),
        ]
        aclient = _FakeStreamingAclient(frames)
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When
        seen = [
            item async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
        ]
        # Then: the partial plus the clamped final built from it.
        assert [item.answer for item in seen] == ["use", "use"]
        assert seen[-1].model_id == "test-model"

    async def test_reraises_transport_errors(self) -> None:
        """Stream failures propagate instead of ending silently."""
        # Given
        aclient = _FakeStreamingAclient(error=ConnectionError("openrouter down"))
        generator = RAGGenerator(config=_config(), aclient=aclient)
        # When / Then
        with pytest.raises(ConnectionError):
            [
                item
                async for item in generator.astream("q", [_hit("docs/a.md", "alpha")])
            ]
