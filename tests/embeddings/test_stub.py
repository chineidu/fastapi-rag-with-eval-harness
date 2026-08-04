from src.embeddings import StubEmbedder


class TestStubEmbedder:
    def test_dim_matches_constructor(self) -> None:
        assert StubEmbedder(dim=16).dim == 16

    def test_default_dim(self) -> None:
        assert StubEmbedder().dim == 8

    def test_model_id_defaults_to_stub(self) -> None:
        assert StubEmbedder().model_id == "stub"

    def test_custom_model_id(self) -> None:
        assert StubEmbedder(model_id="stub-2").model_id == "stub-2"

    def test_returns_one_vector_per_text(self) -> None:
        vectors = StubEmbedder(dim=4).embed_texts(["a", "b", "c"])
        assert len(vectors) == 3

    def test_vector_dimension(self) -> None:
        vectors = StubEmbedder(dim=4).embed_texts(["a", "b", "c"])
        assert all(len(vector) == 4 for vector in vectors)

    def test_vectors_are_in_unit_range(self) -> None:
        vectors = StubEmbedder(dim=16).embed_texts(["a", "b"])
        assert all(-1.0 <= value <= 1.0 for vector in vectors for value in vector)

    def test_deterministic_across_calls(self) -> None:
        embedder = StubEmbedder(dim=8)
        assert embedder.embed_texts(["hello world"]) == embedder.embed_texts(
            ["hello world"]
        )

    def test_different_texts_differ(self) -> None:
        embedder = StubEmbedder(dim=8)
        assert embedder.embed_texts(["cat"]) != embedder.embed_texts(["dog"])

    def test_empty_input_returns_empty_list(self) -> None:
        assert StubEmbedder().embed_texts([]) == []
