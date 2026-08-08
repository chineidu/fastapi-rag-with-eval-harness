from src.embeddings import AbstractEmbedder, ApiEmbedder, LocalEmbedder, StubEmbedder


class TestAbstractEmbedder:
    def test_stub_conforms(self) -> None:
        assert isinstance(StubEmbedder(), AbstractEmbedder)

    def test_local_conforms_without_loading_weights(self) -> None:
        embedder = LocalEmbedder(model_id="BAAI/bge-small-en-v1.5", dim=384)
        assert isinstance(embedder, AbstractEmbedder)

    def test_api_conforms_without_network(self) -> None:
        embedder = ApiEmbedder(model_id="openai/text-embedding-3-small", dim=1536)
        assert isinstance(embedder, AbstractEmbedder)
