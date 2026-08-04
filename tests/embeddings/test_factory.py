import pytest

from src.embeddings import ApiEmbedder, LocalEmbedder, StubEmbedder, make_embedder


class TestMakeEmbedder:
    def test_local_provider(self) -> None:
        embedder = make_embedder("local", dim=384)
        assert isinstance(embedder, LocalEmbedder)

    def test_api_provider(self) -> None:
        embedder = make_embedder("api", dim=1536)
        assert isinstance(embedder, ApiEmbedder)

    def test_stub_provider(self) -> None:
        embedder = make_embedder("stub", dim=4)
        assert isinstance(embedder, StubEmbedder)

    def test_kwargs_are_forwarded(self) -> None:
        embedder = make_embedder("stub", model_id="custom", dim=12)
        assert isinstance(embedder, StubEmbedder)
        assert embedder.model_id == "custom"
        assert embedder.dim == 12

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embeddings provider"):
            make_embedder("nope")
