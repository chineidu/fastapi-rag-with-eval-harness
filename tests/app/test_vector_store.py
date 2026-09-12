"""Tests for QdrantVectorStore using an in-memory Qdrant client."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.app.vector_store import QdrantVectorStore
from src.schemas.containers import QdrantConfig
from src.schemas.retrieval import Chunk, CollectionInfo, SearchHit


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

    def test_ensure_collection_returns_true_on_create(self) -> None:
        """Given no collection, then ensure creates it and reports a rebuild."""
        # Given
        store = _store()
        # When
        rebuilt = store.ensure_collection("m", 4, fingerprint="f1")
        # Then
        assert rebuilt is True

    def test_ensure_collection_is_idempotent(self) -> None:
        """Given a matching model/dim/fingerprint, then a second call skips rebuild."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        rebuilt = store.ensure_collection("m", 4, fingerprint="f1")
        # Then
        assert rebuilt is False
        assert store.chunk_count() == 1

    def test_ensure_collection_rebuilds_on_model_mismatch(self) -> None:
        """Given a model/dim mismatch, then the collection is rebuilt."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        rebuilt = store.ensure_collection("other", 4, fingerprint="f1")
        # Then
        assert rebuilt is True
        assert store.chunk_count() == 0

    def test_ensure_collection_rebuilds_on_fingerprint_mismatch(self) -> None:
        """Given a corpus fingerprint mismatch, then the collection is rebuilt."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        rebuilt = store.ensure_collection("m", 4, fingerprint="f2")
        # Then
        assert rebuilt is True
        assert store.chunk_count() == 0

    def test_ensure_collection_force_rebuilds_matching_collection(self) -> None:
        """Given --force semantics, then a matching collection is still rebuilt."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        rebuilt = store.ensure_collection("m", 4, fingerprint="f1", force=True)
        # Then
        assert rebuilt is True
        assert store.chunk_count() == 0

    def test_describe_missing_collection(self) -> None:
        """Given no collection, then describe reports it as absent."""
        # Given
        store = _store()
        # When
        info = store.describe()
        # Then
        assert isinstance(info, CollectionInfo)
        assert info.exists is False
        assert info.collection == "test"
        assert info.model_id is None
        assert info.dim is None
        assert info.chunk_count == 0

    def test_describe_reports_metadata_and_count(self) -> None:
        """Given an indexed collection, then describe returns its metadata."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "hello")], [[0.1, 0.2, 0.3, 0.4]])
        # When
        info = store.describe()
        # Then
        assert info.exists is True
        assert info.collection == "test"
        assert info.model_id == "m"
        assert info.dim == 4
        assert info.chunk_count == 1

    def test_describe_without_meta_reports_unknown_model(self) -> None:
        """Given a collection created outside the store, then model/dim are None."""
        # Given
        client = QdrantClient(location=":memory:")
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(QdrantConfig(collection="test"), client=client)
        # When
        info = store.describe()
        # Then
        assert info.exists is True
        assert info.model_id is None
        assert info.dim is None
        assert info.chunk_count == 0

    def test_upsert_and_search_returns_metadata(self) -> None:
        """Given upserted chunks, then search returns hits with payload."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
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
        store.ensure_collection("m", 4, fingerprint="f1")
        store.upsert([_chunk("c1", "a.md", "x")], [[0.5, 0.5, 0.5, 0.5]])
        # When
        hits = store.search([0.5, 0.5, 0.5, 0.5], k=5)
        # Then
        assert all(h.chunk_id != "__index_meta__" for h in hits)
        assert store.chunk_count() == 1

    def test_upsert_length_mismatch_raises(self) -> None:
        """Given mismatched chunks and vectors, then ValueError is raised."""
        # Given
        store = _store()
        store.ensure_collection("m", 4, fingerprint="f1")
        # When / Then
        try:
            store.upsert(
                [_chunk("c1", "a.md", "x")],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
