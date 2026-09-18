"""Tests for the document indexer using an in-memory fake vector store."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from src.app.hybrid import TantivyIndex
from src.app.indexer import Indexer, _corpus_fingerprint
from src.embeddings.stub import StubEmbedder
from src.schemas.retrieval import Chunk, CollectionInfo, SearchHit
from src.schemas.types import ChunkStrategyEnum


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build a minimal Chunk for fingerprint tests."""
    return Chunk(
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=1,
        doc_path=chunk_id.split("#")[0],
    )


class FakeVectorStore:
    """Minimal in-memory VectorStore for exercising the Indexer."""

    def __init__(self) -> None:
        """Initialize empty storage for recorded calls."""
        self.ensured: tuple[str, int, str, bool] | None = None
        self.ensure_calls = 0
        self.upserted: list[tuple[list[Chunk], list[list[float]]]] = []
        self._count = 0
        self._fingerprint: str | None = None

    def ensure_collection(
        self, model_id: str, dim: int, *, fingerprint: str, force: bool = False
    ) -> bool:
        """Mimic the store reuse rule and reset storage on rebuild."""
        self.ensured = (model_id, dim, fingerprint, force)
        self.ensure_calls += 1
        if not force and self._count > 0 and self._fingerprint == fingerprint:
            return False
        self._fingerprint = fingerprint
        self._count = 0
        return True

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.upserted.append((chunks, vectors))
        self._count = len(chunks)

    def search(self, query_vector: list[float], k: int) -> list[SearchHit]:
        return []

    def chunk_count(self) -> int:
        return self._count

    def describe(self) -> CollectionInfo:
        return CollectionInfo(
            exists=self._fingerprint is not None,
            collection="fake",
            model_id=None,
            dim=None,
            chunk_count=self._count,
        )


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
        report = indexer.build(tmp_path)
        # Then
        assert report.indexed_chunks > 0
        assert report.skipped is False
        assert store.ensured is not None
        model_id, dim, fingerprint, force = store.ensured
        assert (model_id, dim, force) == ("stub", 8, False)
        assert fingerprint
        assert len(store.upserted) == 1
        chunks, vectors = store.upserted[0]
        assert len(chunks) == report.indexed_chunks
        assert len(vectors) == report.indexed_chunks
        assert all(len(v) == 8 for v in vectors)

    def test_build_skips_unchanged_corpus(self, tmp_path: Path) -> None:
        """Given the same corpus twice, then the second build skips embed/upsert."""
        # Given
        doc = tmp_path / "a.md"
        doc.write_text("FastAPI is a web framework. " * 200, encoding="utf-8")
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store, chunk_size=200, overlap=0)
        first = indexer.build(tmp_path)
        # When
        report = indexer.build(tmp_path)
        # Then
        assert first.skipped is False
        assert report.skipped is True
        assert report.indexed_chunks == 0
        assert store.ensure_calls == 2
        assert len(store.upserted) == 1

    def test_build_reindexes_changed_corpus(self, tmp_path: Path) -> None:
        """Given an edited document, then the next build rebuilds and upserts."""
        # Given
        doc = tmp_path / "a.md"
        doc.write_text("original content " * 100, encoding="utf-8")
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store, chunk_size=100)
        indexer.build(tmp_path)
        # When
        doc.write_text("changed content " * 100, encoding="utf-8")
        report = indexer.build(tmp_path)
        # Then
        assert report.skipped is False
        assert report.indexed_chunks > 0
        assert store.ensure_calls == 2
        assert len(store.upserted) == 2

    def test_build_empty_corpus_returns_empty_report(self, tmp_path: Path) -> None:
        """Given an empty directory, then zero chunks are indexed."""
        # Given
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store)
        # When
        report = indexer.build(tmp_path)
        # Then
        assert report.indexed_chunks == 0
        assert report.skipped is False
        assert store.ensure_calls == 0
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
        assert store.ensured is not None
        assert store.ensured[3] is True

    def test_build_indexes_multiple_roots_in_one_pass(self, tmp_path: Path) -> None:
        """Given two roots, then both are chunked, ensured once, and upserted once."""
        # Given
        md_root = tmp_path / "md"
        md_root.mkdir()
        (md_root / "a.md").write_text("markdown content " * 50, encoding="utf-8")
        py_root = tmp_path / "py"
        py_root.mkdir()
        (py_root / "b.py").write_text("print('hello') " * 50, encoding="utf-8")
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store, chunk_size=100, overlap=0)
        # When
        report = indexer.build([md_root, py_root])
        # Then
        assert report.indexed_chunks > 0
        assert store.ensure_calls == 1
        assert len(store.upserted) == 1
        chunks, _ = store.upserted[0]
        assert {chunk.doc_path for chunk in chunks} == {"a.md", "b.py"}
        assert len(chunks) == report.indexed_chunks

    def test_build_rejects_empty_root_sequence(self, tmp_path: Path) -> None:
        """Given no roots, then build rejects the call."""
        # Given
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store)
        # When / Then
        with pytest.raises(ValueError, match="corpus_roots"):
            indexer.build([])

    def test_build_rejects_bare_string(self, tmp_path: Path) -> None:
        """Given a bare string, then build raises instead of iterating characters."""
        # Given
        store = FakeVectorStore()
        indexer = Indexer(StubEmbedder(dim=8), store)
        # When / Then
        with pytest.raises(TypeError, match="str"):
            indexer.build(cast("Path | Sequence[Path]", str(tmp_path)))

    def test_build_rebuilds_lexical_index(self, tmp_path: Path) -> None:
        """Given a lexical index, then build forwards chunks to it."""
        # Given
        doc = tmp_path / "a.md"
        doc.write_text("FastAPI is a web framework. " * 200, encoding="utf-8")
        store = FakeVectorStore()

        class FakeLexical:
            """Record chunks passed to the lexical build."""

            def __init__(self) -> None:
                """Track received chunks."""
                self.built: list[Chunk] = []

            def build(self, chunks: list[Chunk]) -> None:
                """Record the chunks."""
                self.built = chunks

        lexical = FakeLexical()
        indexer = Indexer(
            StubEmbedder(dim=8),
            store,
            chunk_size=200,
            lexical=cast(TantivyIndex, lexical),
        )
        # When
        report = indexer.build(tmp_path)
        # Then
        assert report.skipped is False
        assert len(lexical.built) == report.indexed_chunks

    def test_build_forwards_chunk_strategy(self, tmp_path: Path) -> None:
        """Given structural strategy, then chunks and fingerprint differ from naive."""
        # Given
        text = "# A\none\n## B\ntwo\n"
        (tmp_path / "doc.md").write_text(text, encoding="utf-8")
        naive_store = FakeVectorStore()
        naive_indexer = Indexer(
            StubEmbedder(dim=8),
            naive_store,
            chunk_size=10,
            overlap=0,
            chunk_strategy=ChunkStrategyEnum.NAIVE,
        )
        structural_store = FakeVectorStore()
        structural_indexer = Indexer(
            StubEmbedder(dim=8),
            structural_store,
            chunk_size=10,
            overlap=0,
            chunk_strategy=ChunkStrategyEnum.STRUCTURAL,
        )
        # When
        naive_report = naive_indexer.build(tmp_path)
        structural_report = structural_indexer.build(tmp_path)
        # Then
        assert naive_report.skipped is False
        assert structural_report.skipped is False
        naive_chunks, _ = naive_store.upserted[0]
        structural_chunks, _ = structural_store.upserted[0]
        assert [chunk.text for chunk in naive_chunks] != [
            chunk.text for chunk in structural_chunks
        ]
        assert naive_store.ensured is not None
        assert structural_store.ensured is not None
        assert naive_store.ensured[2] != structural_store.ensured[2]
        for chunk in structural_chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.text

    def test_build_defaults_to_naive_strategy(self, tmp_path: Path) -> None:
        """Given no strategy, then the indexer chunks like explicit naive."""
        # Given
        (tmp_path / "a.md").write_text("# H\nbody text here\n", encoding="utf-8")
        default_store = FakeVectorStore()
        default_indexer = Indexer(
            StubEmbedder(dim=8), default_store, chunk_size=10, overlap=0
        )
        explicit_store = FakeVectorStore()
        explicit_indexer = Indexer(
            StubEmbedder(dim=8),
            explicit_store,
            chunk_size=10,
            overlap=0,
            chunk_strategy=ChunkStrategyEnum.NAIVE,
        )
        # When
        default_indexer.build(tmp_path)
        explicit_indexer.build(tmp_path)
        # Then
        default_chunks, _ = default_store.upserted[0]
        explicit_chunks, _ = explicit_store.upserted[0]
        assert [chunk.text for chunk in default_chunks] == [
            chunk.text for chunk in explicit_chunks
        ]


class TestCorpusFingerprint:
    """Tests for the corpus fingerprint helper."""

    def test_fingerprint_is_order_independent(self) -> None:
        """Given the same chunks in different order, then fingerprints match."""
        # Given
        chunks = [_chunk("a.md#0000", "alpha"), _chunk("b.md#0000", "beta")]
        # When / Then
        assert _corpus_fingerprint(chunks) == _corpus_fingerprint(
            list(reversed(chunks))
        )

    def test_fingerprint_changes_with_text(self) -> None:
        """Given different chunk text, then fingerprints differ."""
        # Given / When / Then
        assert _corpus_fingerprint([_chunk("a.md#0000", "x")]) != _corpus_fingerprint(
            [_chunk("a.md#0000", "y")]
        )
