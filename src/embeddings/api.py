from typing import Any

import openai

from src.config import app_settings
from src.embeddings.base import AbstractEmbedder

_DEFAULT_MODEL_ID = "openai/text-embedding-3-small"

_KNOWN_DIMS: dict[str, int] = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}


class ApiEmbedder(AbstractEmbedder):
    """Cloud embedder via an OpenAI-compatible ``/embeddings`` endpoint.

    The client is an ``openai.OpenAI`` instance pointed at OpenRouter's
    OpenAI-compatible API, reusing the existing ``OPENROUTER_BASE_URL`` and
    ``OPENROUTER_API_KEY`` settings. A client can be injected for testing.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        batch_size: int = 32,
        dim: int | None = None,
        client: Any = None,
    ) -> None:
        self.model_id = model_id
        self._batch_size = batch_size
        self._dim = dim if dim is not None else _KNOWN_DIMS.get(model_id)
        self._client = client or self._get_sync_client()
        self._async_client: openai.AsyncOpenAI | None = None

    def _get_sync_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=app_settings.OPENROUTER_BASE_URL,
            api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
        )

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model_id, input=batch)
        return [list(map(float, item.embedding)) for item in response.data]

    def _get_async_client(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(
                base_url=app_settings.OPENROUTER_BASE_URL,
                api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
            )
        return self._async_client

    async def _aembed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._get_async_client().embeddings.create(
            model=self.model_id, input=batch
        )
        return [list(map(float, item.embedding)) for item in response.data]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors via the async client.

        Uses ``AsyncOpenAI`` for genuinely non-blocking I/O rather than the
        thread-offloaded default from the base class. Batching behaviour is
        identical to :meth:`embed_texts`.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed. May be empty.

        Returns
        -------
        list[list[float]]
            One vector per input text, in the same order.
            Each vector has length ``self.dim``.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                await self._aembed_batch(texts[start : start + self._batch_size])
            )
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors
