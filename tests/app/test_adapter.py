"""Tests for LocalRetriever chunk-to-document deduplication (ADR-0021)."""

from types import SimpleNamespace
from typing import cast

import pytest
from qdrant_client import QdrantClient

from src.app.adapter import LocalRetriever
from src.app.hybrid import TantivyIndex
from src.app.vector_store import QdrantVectorStore
from src.embeddings.stub import StubEmbedder
from src.schemas.containers import QdrantConfig
from src.schemas.models import AppConfig
from src.schemas.retrieval import Chunk, CollectionInfo, SearchHit


def _test_config(*, hybrid_enabled: bool = False) -> SimpleNamespace:
    """App config stub with hybrid retrieval disabled by default."""
    return SimpleNamespace(
        embeddings_config=SimpleNamespace(),
        indexer_config=SimpleNamespace(chunk_size=2000, overlap=0),
        retriever_config=SimpleNamespace(
            overfetch_factor=5,
            hybrid_enabled=hybrid_enabled,
            sparse_k=50,
            rrf_k=60,
            dense_weight=1.0,
            sparse_weight=1.0,
            tantivy_index_dir="",
        ),
    )


class FakeStore:
    """VectorStore stub returning a fixed, ordered hit list."""

    def __init__(self, hits: list[SearchHit]) -> None:
        """Store the fixed hit list."""
        self._hits = hits
        self.last_limit: int | None = None

    def ensure_collection(
        self, model_id: str, dim: int, *, fingerprint: str, force: bool = False
    ) -> bool:
        return True

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        pass

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        self.last_limit = k
        return self._hits[:k]

    def chunk_count(self) -> int:
        return len(self._hits)

    def describe(self) -> CollectionInfo:
        return CollectionInfo(
            exists=True,
            collection="fake",
            model_id=None,
            dim=None,
            chunk_count=len(self._hits),
        )


class ExplodingStore(FakeStore):
    """VectorStore stub whose search always fails."""

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        raise ConnectionError("qdrant unavailable")


class FakeLexical:
    """TantivyIndex stub returning a fixed sparse hit list."""

    def __init__(self, hits: list[SearchHit]) -> None:
        """Store the fixed hit list."""
        self._hits = hits

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the fixed hits, capped at k."""
        return self._hits[:k]


def _hit(doc_path: str, score: float, chunk_index: int = 0) -> SearchHit:
    """Build one SearchHit for the given document and score."""
    return SearchHit(
        chunk_id=f"{doc_path}#{chunk_index:04d}",
        doc_path=doc_path,
        chunk_index=chunk_index,
        text=f"text {chunk_index}",
        score=score,
    )


def _retriever(store: FakeStore, factor: int) -> LocalRetriever:
    """Build a LocalRetriever with injected fakes."""
    return LocalRetriever(
        embedder=StubEmbedder(),
        store=store,
        overfetch_factor=factor,
        config=cast(AppConfig, _test_config()),
    )


class TestLocalRetriever:
    def test_deduplicates_documents_keeping_best_score(self) -> None:
        """Repeated chunks collapse to one document at its best score."""
        # Given
        store = FakeStore(
            [
                _hit("docs/a.md", 0.9, 0),
                _hit("docs/a.md", 0.8, 1),
                _hit("docs/b.md", 0.7, 0),
                _hit("docs/b.md", 0.6, 1),
                _hit("docs/c.md", 0.5, 0),
            ]
        )
        retriever = _retriever(store, factor=5)
        # When
        result = retriever.retrieve("query", k=10)
        # Then
        assert [(doc.doc_path, doc.score) for doc in result.documents] == [
            ("docs/a.md", 0.9),
            ("docs/b.md", 0.7),
            ("docs/c.md", 0.5),
        ]
        assert store.last_limit == 50

    def test_unsorted_hits_still_keep_best_score_per_document(self) -> None:
        """Backend order is not trusted; docs rank by their best chunk score."""
        # Given: hits for the same documents arrive out of score order
        store = FakeStore(
            [
                _hit("docs/a.md", 0.4, 0),
                _hit("docs/b.md", 0.9, 0),
                _hit("docs/a.md", 0.8, 1),
                _hit("docs/c.md", 0.7, 0),
                _hit("docs/b.md", 0.6, 1),
            ]
        )
        retriever = _retriever(store, factor=5)
        # When
        result = retriever.retrieve("query", k=3)
        # Then
        assert [(doc.doc_path, doc.score) for doc in result.documents] == [
            ("docs/b.md", 0.9),
            ("docs/a.md", 0.8),
            ("docs/c.md", 0.7),
        ]

    def test_factor_one_undercounts_docs_when_chunks_cluster(self) -> None:
        """Without over-fetch, clustered chunks cap the document count."""
        # Given
        store = FakeStore(
            [
                _hit("docs/a.md", 0.9, 0),
                _hit("docs/a.md", 0.8, 1),
                _hit("docs/a.md", 0.7, 2),
                _hit("docs/b.md", 0.6, 0),
                _hit("docs/c.md", 0.5, 0),
            ]
        )
        retriever = _retriever(store, factor=1)
        # When
        result = retriever.retrieve("query", k=3)
        # Then
        assert [doc.doc_path for doc in result.documents] == ["docs/a.md"]
        assert store.last_limit == 3

    def test_overfetch_fills_document_slots(self) -> None:
        """Over-fetching surfaces documents ranked below the k-th chunk."""
        # Given
        store = FakeStore(
            [
                _hit("docs/a.md", 0.9, 0),
                _hit("docs/a.md", 0.8, 1),
                _hit("docs/a.md", 0.7, 2),
                _hit("docs/b.md", 0.6, 0),
                _hit("docs/c.md", 0.5, 0),
            ]
        )
        retriever = _retriever(store, factor=5)
        # When
        result = retriever.retrieve("query", k=3)
        # Then
        assert [doc.doc_path for doc in result.documents] == [
            "docs/a.md",
            "docs/b.md",
            "docs/c.md",
        ]
        assert store.last_limit == 15

    def test_truncates_to_k_documents(self) -> None:
        """More unique documents than k are truncated to the best k."""
        # Given
        store = FakeStore(
            [
                _hit("docs/a.md", 0.9, 0),
                _hit("docs/b.md", 0.8, 0),
                _hit("docs/c.md", 0.7, 0),
                _hit("docs/d.md", 0.6, 0),
            ]
        )
        retriever = _retriever(store, factor=5)
        # When
        result = retriever.retrieve("query", k=2)
        # Then
        assert [doc.doc_path for doc in result.documents] == [
            "docs/a.md",
            "docs/b.md",
        ]

    def test_metadata_reports_retrieval_shape(self) -> None:
        """Metadata records the window and the fetched/returned counts."""
        # Given
        store = FakeStore([_hit("docs/a.md", 0.9, 0), _hit("docs/a.md", 0.8, 1)])
        retriever = _retriever(store, factor=4)
        # When
        result = retriever.retrieve("query", k=5)
        # Then
        assert result.metadata["model_id"] == "stub"
        assert result.metadata["overfetch_factor"] == 4
        assert result.metadata["chunk_limit"] == 20
        assert result.metadata["chunks_fetched"] == 2
        assert result.metadata["docs_returned"] == 1
        assert result.metadata["k"] == 5
        assert "chunk_size" in result.metadata
        assert "overlap" in result.metadata

    def test_empty_store_returns_no_documents(self) -> None:
        """An empty index yields an empty, well-formed result."""
        # Given
        retriever = _retriever(FakeStore([]), factor=5)
        # When
        result = retriever.retrieve("query", k=10)
        # Then
        assert result.documents == []
        assert result.metadata["chunks_fetched"] == 0
        assert result.metadata["docs_returned"] == 0

    def test_rejects_non_positive_overfetch_factor(self) -> None:
        """A factor below 1 is a configuration error."""
        # Given / When / Then
        with pytest.raises(ValueError, match="overfetch_factor"):
            LocalRetriever(
                embedder=StubEmbedder(), store=FakeStore([]), overfetch_factor=0
            )

    def test_rejects_non_positive_k(self) -> None:
        """A non-positive k is a caller error."""
        # Given
        retriever = _retriever(FakeStore([]), factor=5)
        # When / Then
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.retrieve("query", k=0)

    def test_store_error_propagates(self) -> None:
        """Store failures reach the harness for status classification."""
        # Given
        retriever = _retriever(ExplodingStore([]), factor=5)
        # When / Then
        with pytest.raises(ConnectionError):
            retriever.retrieve("query", k=10)

    def test_hybrid_fuses_sparse_hits(self) -> None:
        """Given hybrid enabled, then sparse-only docs enter the result."""
        # Given
        store = FakeStore([_hit("docs/a.md", 0.9, 0)])
        lexical = cast(TantivyIndex, FakeLexical([_hit("docs/b.md", 8.0, 0)]))
        retriever = LocalRetriever(
            embedder=StubEmbedder(),
            store=store,
            overfetch_factor=5,
            config=cast(AppConfig, _test_config(hybrid_enabled=True)),
            lexical=lexical,
        )
        # When
        result = retriever.retrieve("query", k=10)
        # Then
        assert {doc.doc_path for doc in result.documents} == {
            "docs/a.md",
            "docs/b.md",
        }
        assert result.metadata["hybrid_enabled"] is True
        assert result.metadata["sparse_fetched"] == 1

    def test_hybrid_without_lexical_warns_and_uses_dense(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hybrid without an index warns and falls back to dense hits."""
        # Given (__init__ auto-builds when enabled, so force the missing case)
        store = FakeStore([_hit("docs/a.md", 0.9, 0)])
        retriever = LocalRetriever(
            embedder=StubEmbedder(),
            store=store,
            overfetch_factor=5,
            config=cast(AppConfig, _test_config(hybrid_enabled=True)),
            lexical=None,
        )
        retriever._lexical = None
        # When
        with caplog.at_level("WARNING", logger="src.app.adapter"):
            result = retriever.retrieve("query", k=10)
        # Then
        assert [doc.doc_path for doc in result.documents] == ["docs/a.md"]
        assert result.metadata["sparse_fetched"] == 0
        assert any("lexical index" in record.message for record in caplog.records)

    def test_generate_is_deferred(self) -> None:
        """Generation is out of scope until ADR-0004's deferred phase."""
        # Given
        retriever = _retriever(FakeStore([]), factor=5)
        # When / Then
        with pytest.raises(NotImplementedError):
            retriever.generate("query", ["docs/a.md"])


class TestLocalRetrieverWithQdrant:
    def test_retrieves_documents_from_indexed_chunks(self) -> None:
        """The adapter works end-to-end against a real in-memory store."""
        # Given
        embedder = StubEmbedder()
        store = QdrantVectorStore(
            QdrantConfig(collection="test"), client=QdrantClient(location=":memory:")
        )
        chunks = [
            Chunk(
                chunk_id=f"docs/{name}.md#{index:04d}",
                chunk_index=index,
                text=f"{name} content {index}",
                start_char=0,
                end_char=10,
                token_count=3,
                doc_path=f"docs/{name}.md",
            )
            for name, count in (("a", 3), ("b", 2))
            for index in range(count)
        ]
        store.ensure_collection(embedder.model_id, embedder.dim, fingerprint="test")
        store.upsert(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
        retriever = LocalRetriever(
            embedder=embedder,
            store=store,
            overfetch_factor=5,
            config=cast(AppConfig, _test_config()),
        )
        # When
        result = retriever.retrieve("content", k=10)
        # Then
        assert {doc.doc_path for doc in result.documents} == {
            "docs/a.md",
            "docs/b.md",
        }
        assert result.metadata["chunks_fetched"] == 5
