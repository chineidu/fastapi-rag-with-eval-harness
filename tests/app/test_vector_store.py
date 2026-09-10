"""Tests for QdrantVectorStore using an in-memory Qdrant client."""

from qdrant_client import QdrantClient

from src.app.vector_store import QdrantVectorStore
from src.schemas.containers import QdrantConfig
from src.schemas.retrieval import Chunk, SearchHit


def _store() -> QdrantVectorStore:
    """Build a QdrantVectorStore backed by an in-memory client."""
    client = QdrantClient(location=":memory:")
    return QdrantVectorStore(QdrantConfig(collection="test"), client=client)


def _chunk(chunk_id: str, doc_path: str, text: str) -> Chunk:
    """Build a minimal Chunk for tests."""
    return Chunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
        doc_path=doc_path,
    )


class TestQdrantVectorStore:
    """Tests for the Qdrant-backed vector store."""

    def test_ensure_collection_is_idempotent(self) -> None:
        """Given a matching model/dim, then a second call skips rebuild."""
        # Given
        store = _store()
        store.ensure_collection("m", 4)
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        store.ensure_collection("m", 4)
        # Then
        assert store.count() == 1

    def test_ensure_collection_rebuilds_on_mismatch(self) -> None:
        """Given a model/dim mismatch, then the collection is rebuilt."""
        # Given
        store = _store()
        store.ensure_collection("m", 4)
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        store.ensure_collection("other", 4)
        # Then
        assert store.count() == 0

    def test_upsert_and_search_returns_metadata(self) -> None:
        """Given upserted chunks, then search returns hits with payload."""
        # Given
        store = _store()
        store.ensure_collection("m", 4)
        store.upsert(
            [
                _chunk("c1", "a.md", "fastapi routing"),
                _chunk("c2", "b.md", "sqlalchemy models"),
            ],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        # When
        hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        # Then
        assert len(hits) == 2
        assert isinstance(hits[0], SearchHit)
        assert hits[0].chunk_id == "c1"
        assert hits[0].doc_path == "a.md"
        assert hits[0].text == "fastapi routing"
        assert hits[0].score > hits[1].score

    def test_search_excludes_meta_point(self) -> None:
        """Given a query, then the sentinel meta point is never returned."""
        # Given
        store = _store()
        store.ensure_collection("m", 4)
        store.upsert([_chunk("c1", "a.md", "x")], [[0.5, 0.5, 0.5, 0.5]])
        # When
        hits = store.search([0.5, 0.5, 0.5, 0.5], k=5)
        # Then
        assert all(h.chunk_id != "__index_meta__" for h in hits)
        assert store.count() == 1

    def test_upsert_length_mismatch_raises(self) -> None:
        """Given mismatched chunks and vectors, then ValueError is raised."""
        # Given
        store = _store()
        store.ensure_collection("m", 4)
        # When / Then
        try:
            store.upsert(
                [_chunk("c1", "a.md", "x")],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
