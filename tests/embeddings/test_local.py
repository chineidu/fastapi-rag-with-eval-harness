import pytest

from src.embeddings import LocalEmbedder


@pytest.mark.slow
class TestLocalEmbedderReal:
    """Downloads bge-small-en-v1.5 weights (~67MB) on first run; use -m slow."""

    def test_embeds_text(self) -> None:
        embedder = LocalEmbedder(model_id="BAAI/bge-small-en-v1.5")
        vectors = embedder.embed_texts(
            ["The quick brown fox", "jumps over the lazy dog"]
        )
        assert len(vectors) == 2
        assert all(len(vector) == 384 for vector in vectors)

    def test_deterministic(self) -> None:
        embedder = LocalEmbedder(model_id="BAAI/bge-small-en-v1.5")
        assert embedder.embed_texts(["hello world"]) == embedder.embed_texts(
            ["hello world"]
        )

    def test_dim_resolves_from_model(self) -> None:
        embedder = LocalEmbedder(model_id="BAAI/bge-small-en-v1.5")
        assert embedder.dim == 384
