"""Abstract base class shared by all embedding implementations."""

import asyncio
from abc import ABC, abstractmethod


class AbstractEmbedder(ABC):
    """Abstract base for text-embedding implementations.

    Subclasses must expose the active model id, the embedding dimension,
    and a batched ``embed_texts`` method. The harness and the labeling
    pipeline only ever depend on this interface, so the local, API, and
    stub embedders are interchangeable.
    """

    model_id: str
    _batch_size: int = 32
    _dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding dimensionality produced by :meth:`embed_texts`."""
        if self._dim is None:
            self._dim = len(self.embed_texts(["probe"])[0])
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed. May be empty.

        Returns
        -------
        list[list[float]]
            One vector per input text, in the same order. Each vector has
            length ``self.dim``.

        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors without blocking the event loop.

        Delegates to :meth:`embed_texts` via ``asyncio.to_thread``, so the
        batching and dimensionality-lazy-initialisation behaviour is identical.

        Subclasses that provide a genuinely asynchronous embedding backend
        should override this method directly rather than relying on the
        thread-offloaded default.

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
        return await asyncio.to_thread(self.embed_texts, texts)

    @abstractmethod
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors."""
