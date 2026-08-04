from typing import Any

import openai

from src.config import app_settings

_DEFAULT_MODEL_ID = "openai/text-embedding-3-small"

_KNOWN_DIMS: dict[str, int] = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}


class ApiEmbedder:
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
        self._client = client or openai.OpenAI(
            base_url=app_settings.OPENROUTER_BASE_URL,
            api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
        )

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_texts(["probe"])[0])
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self.model_id, input=batch)
            vectors.extend(list(map(float, item.embedding)) for item in response.data)
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors
