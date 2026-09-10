"""Tests for the document indexer using an in-memory fake vector store."""

from pathlib import Path

from src.app.indexer import Indexer
from src.embeddings.stub import StubEmbedder
from src.schemas.retrieval import Chunk, SearchHit


class FakeVectorStore:
    """Minimal in-memory VectorStore for exercising the Indexer."""

    def __init__(self) -> None:
        """Initialize empty storage for recorded calls."""
        self.ensured: tuple[str, int, bool] | None = None
        self.upserted: list[tuple[list[Chunk], list[list[float]]]] = []
        self._count = 0

    def ensure_collection(
        self, model_id: str, dim: int, *, force: bool = False
    ) -> None:
        self.ensured = (model_id, dim, force)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.upserted.append((chunks, vectors))
        self._count = len(chunks)

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        return []

    def count(self) -> int:
        return self._count


class TestIndexer:
    """Tests for Indexer.build."""

    def test_build_indexes_chunks(self, tmp_path: Path) -> None:
        """Given a corpus, then chunks are embedded and upserted."""
        # Given
        doc = tmp_path / "a.md"
        doc.write_text("FastAPI is a web framework. " * 200, encoding="utf-8")
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store, chunk_size=200, overlap=0)
        # When
        count = indexer.build(tmp_path)
        # Then
        assert count > 0
        assert store.ensured == ("stub", 8, False)
        assert len(store.upserted) == 1
        chunks, vectors = store.upserted[0]
        assert len(chunks) == count
        assert len(vectors) == count
        assert all(len(v) == 8 for v in vectors)

    def test_build_empty_corpus_returns_zero(self, tmp_path: Path) -> None:
        """Given an empty directory, then zero chunks are indexed."""
        # Given
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store)
        # When
        count = indexer.build(tmp_path)
        # Then
        assert count == 0
        assert store.upserted == []

    def test_build_passes_force_flag(self, tmp_path: Path) -> None:
        """Given force=True, then ensure_collection receives force."""
        # Given
        doc = tmp_path / "a.md"
        doc.write_text("content " * 100, encoding="utf-8")
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store, chunk_size=100)
        # When
        indexer.build(tmp_path, force=True)
        # Then
        assert store.ensured == ("stub", 8, True)
