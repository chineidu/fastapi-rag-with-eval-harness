"""Few-shot grounded generation over retrieved chunk texts."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import instructor
import openai

from src.config import app_settings, load_app_config
from src.prompts.generation import GENERATE_SYSTEM_PROMPT, format_generation_context
from src.schemas.containers import RAGLLMConfig
from src.schemas.generation import GeneratedAnswer, GenerationPrompt
from src.schemas.models import AppConfig
from src.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)

__all__ = ["RAGGenerator"]


class RAGGenerator:
    """Generate grounded answers from best-chunk texts via OpenRouter.

    Contexts are the fused ``SearchHit`` chunk hits from retrieval, so no
    disk reload is needed. Prompt assembly is exposed via
    :meth:`build_prompt` for unit testing without network calls.
    """

    def __init__(
        self, *, config: AppConfig | None = None, aclient: Any | None = None
    ) -> None:
        """Configure the generator.

        Parameters
        ----------
        config : AppConfig | None
            Application config. Loaded from the bundled YAML when ``None``.
        aclient : Any | None
            Injectable instructor async client (tests). Built from
            ``rag_config.llm`` and ``OPENROUTER_*`` settings when ``None``.

        """
        cfg = config or load_app_config()
        llm: RAGLLMConfig = cfg.rag_config.llm
        self._llm = llm
        self._model_id = llm.model_id
        if aclient is not None:
            self._aclient = aclient
        else:
            openai_client = openai.AsyncOpenAI(
                base_url=app_settings.OPENROUTER_BASE_URL,
                api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
                timeout=llm.timeout_seconds,
                max_retries=llm.max_retries,
            )
            self._aclient = instructor.from_openai(openai_client)

    async def agenerate(self, query: str, contexts: list[SearchHit]) -> GeneratedAnswer:
        """Generate a structured answer from chunk contexts.

        Parameters
        ----------
        query : str
            The user question.
        contexts : list[SearchHit]
            Ranked chunk hits supplying answer grounding.

        Returns
        -------
        GeneratedAnswer
            Answer text with citations clamped to context paths, the
            configured model id, and the grounding flag.

        Raises
        ------
        Exception
            Transport and LLM errors propagate after logging; only a
            model-unknown yields the abstention answer, never an error.

        """
        prompt = self.build_prompt(query, contexts)
        try:
            response = await self._aclient.completions.create(
                model=self._model_id,
                response_model=GeneratedAnswer,
                temperature=self._llm.temperature,
                seed=self._llm.seed,
                max_tokens=self._llm.max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
            )
        except Exception:
            logger.exception("Generation failed for query %r", query[:120])
            raise
        return self._clamp_citations(response, contexts)

    async def astream(
        self, query: str, contexts: list[SearchHit]
    ) -> AsyncIterator[GeneratedAnswer]:
        """Stream partial structured answers from chunk contexts.

        Parameters
        ----------
        query : str
            The user question.
        contexts : list[SearchHit]
            Ranked chunk hits supplying answer grounding.

        Yields
        ------
        GeneratedAnswer
            Growing partial snapshots followed by the citation-clamped
            final answer. Mid-stream snapshots may carry incomplete
            citations; only the final yield is clamped.

        Raises
        ------
        Exception
            Transport and LLM errors propagate after logging.

        """
        prompt = self.build_prompt(query, contexts)
        try:
            partials = self._aclient.chat.completions.create_partial(
                model=self._model_id,
                response_model=GeneratedAnswer,
                temperature=self._llm.temperature,
                seed=self._llm.seed,
                max_tokens=self._llm.max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
            )
            last: GeneratedAnswer | None = None
            async for partial in partials:
                yield partial
                last = partial
        except Exception:
            logger.exception("Generation stream failed for query %r", query[:120])
            raise
        if last is not None:
            yield self._clamp_citations(last, contexts)

    def _clamp_citations(
        self, response: GeneratedAnswer, contexts: list[SearchHit]
    ) -> GeneratedAnswer:
        """Drop citations outside the context paths, preserving model order.

        Parameters
        ----------
        response : GeneratedAnswer
            Answer whose citations need clamping.
        contexts : list[SearchHit]
            Ranked chunk hits defining the valid paths.

        Returns
        -------
        GeneratedAnswer
            Copy with unknown citations removed and the model id taken
            from config, not the LLM echo.

        """
        # Light clamp: keep model order with duplicates, drop unknown paths.
        valid: set[str] = {hit.doc_path for hit in contexts}
        cites: list[str] = response.citations or []
        kept: list[str] = [cite for cite in cites if cite in valid]
        if len(kept) != len(cites):
            logger.warning("Dropped %d unknown citations", len(cites) - len(kept))
        # Config is the source of truth for model identity, not the LLM echo.
        return GeneratedAnswer(
            answer=response.answer,
            citations=kept,
            model_id=self._model_id,
            grounded=response.grounded,
        )

    def build_prompt(self, query: str, contexts: list[SearchHit]) -> GenerationPrompt:
        """Assemble the few-shot grounded prompt pair.

        Parameters
        ----------
        query : str
            The user question.
        contexts : list[SearchHit]
            Ranked chunk hits formatted into the user message.

        Returns
        -------
        GenerationPrompt
            System and user strings ready for the chat completion call.
            Documents precede the question per long-context guidance.

        """
        user = (
            f"{format_generation_context(contexts)}\n\n<question>\n{query}\n</question>"
        )
        return GenerationPrompt(system=GENERATE_SYSTEM_PROMPT, user=user)
