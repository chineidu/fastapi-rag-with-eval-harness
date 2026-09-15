"""Few-shot grounded generation over retrieved chunk texts."""

import logging
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
        # Light clamp: keep model order with duplicates, drop unknown paths.
        valid: set[str] = {hit.doc_path for hit in contexts}
        kept: list[str] = [cite for cite in response.citations if cite in valid]
        if len(kept) != len(response.citations):
            logger.warning(
                "Dropped %d unknown citations", len(response.citations) - len(kept)
            )
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
