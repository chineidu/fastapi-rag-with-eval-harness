from types import SimpleNamespace

from src.embeddings import ApiEmbedder


class FakeEmbeddingsApi:
    """Records calls and returns canned embeddings without any network."""

    def __init__(self, calls: list[tuple[str, list[str]]]) -> None:
        """Record every request into the shared calls list."""
        self._calls = calls

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        self._calls.append((model, list(input)))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index), float(index) + 1.0])
                for index in range(len(input))
            ]
        )


class FakeClient:
    """Minimal stand-in for ``openai.OpenAI`` exposing ``embeddings.create``."""

    def __init__(self) -> None:
        """Build a fake client with an empty calls list."""
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = FakeEmbeddingsApi(self.calls)


class TestApiEmbedder:
    def test_embeds_single_text(self) -> None:
        client = FakeClient()
        embedder = ApiEmbedder(
            model_id="openai/text-embedding-3-small", dim=1536, client=client
        )
        vectors = embedder.embed_texts(["hello"])
        assert len(vectors) == 1
        assert vectors[0] == [0.0, 1.0]
        assert client.calls == [("openai/text-embedding-3-small", ["hello"])]

    def test_batches_large_inputs(self) -> None:
        client = FakeClient()
        embedder = ApiEmbedder(model_id="m", batch_size=2, dim=2, client=client)
        texts = ["a", "b", "c", "d", "e"]
        vectors = embedder.embed_texts(texts)
        assert len(vectors) == 5
        assert len(client.calls) == 3
        assert client.calls[0] == ("m", ["a", "b"])
        assert client.calls[1] == ("m", ["c", "d"])
        assert client.calls[2] == ("m", ["e"])

    def test_vectors_keep_input_order(self) -> None:
        client = FakeClient()
        embedder = ApiEmbedder(model_id="m", batch_size=1, dim=2, client=client)
        vectors = embedder.embed_texts(["first", "second"])
        assert vectors == [[0.0, 1.0], [0.0, 1.0]]

    def test_dim_known_from_table_without_embedding(self) -> None:
        embedder = ApiEmbedder(model_id="openai/text-embedding-3-small")
        assert embedder.dim == 1536

    def test_dim_learned_from_first_embedding(self) -> None:
        client = FakeClient()
        embedder = ApiEmbedder(model_id="m", dim=None, client=client)
        assert embedder.dim == 2
        assert client.calls == [("m", ["probe"])]

    def test_empty_input_makes_no_calls(self) -> None:
        client = FakeClient()
        embedder = ApiEmbedder(model_id="m", dim=2, client=client)
        assert embedder.embed_texts([]) == []
        assert client.calls == []
