"""Tests for the eval_harness.adapter module."""

from src.schemas.harness import RetrievalResult, RetrievedDocument


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_creates_with_documents(self) -> None:
        """Given a list of documents, then the result stores them."""
        # Given
        docs = [RetrievedDocument("doc/a.md", 0.9), RetrievedDocument("doc/b.md", 0.7)]

        # When
        result = RetrievalResult(documents=docs)

        # Then
        assert result.documents == docs
        assert result.metadata == {}

    def test_creates_with_metadata(self) -> None:
        """Given metadata dict, then the result stores it alongside documents."""
        # Given
        docs = [RetrievedDocument("doc/a.md", 0.9)]
        meta: dict[str, object] = {"latency_ms": 12.5, "model": "bge-small"}

        # When
        result = RetrievalResult(documents=docs, metadata=meta)

        # Then
        assert result.metadata == meta

    def test_empty_documents(self) -> None:
        """Given no documents, then the result has an empty list."""
        # Given / When
        result = RetrievalResult(documents=[])

        # Then
        assert result.documents == []

    def test_slots_allow_reassignment(self) -> None:
        """Given a result with slots, then documents can be reassigned on instances."""
        # Given
        result = RetrievalResult(documents=[RetrievedDocument("a", 1.0)])

        # When
        result.documents = [RetrievedDocument("b", 0.5)]

        # Then
        assert result.documents == [RetrievedDocument("b", 0.5)]
