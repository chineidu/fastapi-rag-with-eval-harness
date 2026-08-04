from src.embeddings import ApiEmbedder, EmbedderPort, LocalEmbedder, StubEmbedder


class TestEmbedderPort:
    def test_stub_conforms(self) -> None:
        assert isinstance(StubEmbedder(), EmbedderPort)

    def test_local_conforms_without_loading_weights(self) -> None:
        embedder = LocalEmbedder(model_id="BAAI/bge-small-en-v1.5", dim=384)
        assert isinstance(embedder, EmbedderPort)

    def test_api_conforms_without_network(self) -> None:
        embedder = ApiEmbedder(model_id="openai/text-embedding-3-small", dim=1536)
        assert isinstance(embedder, EmbedderPort)
