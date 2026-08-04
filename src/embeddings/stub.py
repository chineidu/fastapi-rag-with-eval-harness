import hashlib
import random


class StubEmbedder:
    """Deterministic embedder for tests and offline pipelines.

    Vectors are seeded from the SHA-256 of each text, so identical inputs
    always produce identical vectors. Requires no network access and no
    model weights.
    """

    def __init__(self, model_id: str = "stub", dim: int = 8) -> None:
        self.model_id = model_id
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            seed = int.from_bytes(digest, byteorder="big")
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(self.dim)])
        return vectors
