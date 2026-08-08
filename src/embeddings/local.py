from fastembed import TextEmbedding

from src.embeddings.base import AbstractEmbedder

_DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class LocalEmbedder(AbstractEmbedder):
    """Local ONNX embedder backed by fastembed.

    The model is loaded lazily on the first ``embed_texts`` call, so
    constructing a ``LocalEmbedder`` never downloads weights or touches the
    network. The dimension is resolved from a known-models table at
    construction time; unknown models fall back to probing the first
    embedding.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        cache_dir: str | None = None,
        batch_size: int = 32,
        dim: int | None = None,
    ) -> None:
        self.model_id = model_id
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._dim = dim if dim is not None else _KNOWN_DIMS.get(model_id)
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        """Lazily load the model."""
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_id, cache_dir=self._cache_dir
            )
        return self._model

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        # model.embed(batch) yields a numpy array per text; e.g. for
        # batch=["a", "b"] it yields two arrays of shape (384,), which
        # are converted to two lists of 384 floats.
        return [list(map(float, vector)) for vector in model.embed(batch)]
