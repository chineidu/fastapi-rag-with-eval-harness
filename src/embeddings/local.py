from fastembed import TextEmbedding

_DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class LocalEmbedder:
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

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_texts(["probe"])[0])
        return self._dim

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.model_id, cache_dir=self._cache_dir
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            for vector in model.embed(batch):
                vectors.append(list(map(float, vector)))
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors
