"""Tests for the rag-index CLI."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from src.app import cli as cli_module
from src.schemas.retrieval import CollectionInfo, IndexReport

RUNNER = CliRunner()

BuildCall = tuple[list[Path], bool]


@pytest.fixture
def cli_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[BuildCall]]:
    """Point the CLI at a fake repo root and stub indexing.

    Returns the fake repo root and the list of recorded ``build()`` calls.
    """
    # Lay out both default corpus roots under a fake repo root.
    md_root = tmp_path / "docs" / "fastapi" / "docs" / "en" / "docs"
    py_root = tmp_path / "docs" / "fastapi" / "docs_src"
    md_root.mkdir(parents=True)
    py_root.mkdir(parents=True)

    # Stub config, embedder, and store so no real services are touched.
    cfg = SimpleNamespace(
        embeddings_config=SimpleNamespace(),
        indexer_config=SimpleNamespace(
            chunk_size=100,
            overlap=0,
            qdrant=SimpleNamespace(collection="test_docs"),
            backend=SimpleNamespace(value="qdrant"),
        ),
    )
    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: cfg)
    monkeypatch.setattr(cli_module, "get_embedder", lambda config: object())
    monkeypatch.setattr(cli_module, "get_vector_store", lambda config: object())

    # Record build() calls instead of chunking and embedding.
    calls: list[BuildCall] = []

    class RecordingIndexer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def build(
            self, corpus_roots: list[Path], *, force: bool = False
        ) -> IndexReport:
            calls.append((list(corpus_roots), force))
            return IndexReport(indexed_chunks=7, skipped=False)

    monkeypatch.setattr(cli_module, "Indexer", RecordingIndexer)
    return tmp_path, calls


def _stub_indexer(
    monkeypatch: pytest.MonkeyPatch,
    report: IndexReport,
    store: Any = None,
) -> None:
    """Replace the indexer (and optionally the store) with fixed fakes."""

    class StubIndexer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def build(
            self, corpus_roots: list[Path], *, force: bool = False
        ) -> IndexReport:
            return report

    monkeypatch.setattr(cli_module, "Indexer", StubIndexer)
    if store is not None:
        monkeypatch.setattr(cli_module, "get_vector_store", lambda config: store)


class TestBuildCommand:
    """Tests for the single-command rag-index build surface."""

    def test_defaults_to_both_corpus_roots(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given no --corpus, then both default corpus roots are indexed."""
        # Given
        root, calls = cli_env
        # When
        result = RUNNER.invoke(cli_module.app, ["build"])
        # Then
        assert result.exit_code == 0
        assert calls == [
            (
                [
                    root / "docs" / "fastapi" / "docs" / "en" / "docs",
                    root / "docs" / "fastapi" / "docs_src",
                ],
                False,
            )
        ]
        assert "Indexed 7 chunks into collection test_docs" in result.output

    def test_repeated_corpus_flags_are_collected(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given repeated --corpus flags, then all paths are indexed in order."""
        # Given
        root, calls = cli_env
        extra = root / "extra"
        extra.mkdir()
        # When
        result = RUNNER.invoke(
            cli_module.app,
            [
                "build",
                "--corpus",
                "docs/fastapi/docs_src",
                "--corpus",
                "extra",
            ],
        )
        # Then
        assert result.exit_code == 0
        assert calls == [([root / "docs" / "fastapi" / "docs_src", extra], False)]

    def test_missing_corpus_path_exits_two(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given a missing corpus path, then the CLI exits with code 2."""
        # Given
        _, calls = cli_env
        # When
        result = RUNNER.invoke(cli_module.app, ["build", "--corpus", "does/not/exist"])
        # Then
        assert result.exit_code == 2
        assert "do not exist" in result.output
        assert calls == []

    def test_empty_corpus_path_exits_two(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given an empty --corpus value, then the CLI exits with code 2."""
        # Given
        _, calls = cli_env
        # When
        result = RUNNER.invoke(cli_module.app, ["build", "--corpus", ""])
        # Then
        assert result.exit_code == 2
        assert "must not be empty" in result.output
        assert calls == []

    def test_force_flag_is_forwarded(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given --force, then build receives force=True."""
        # Given
        _, calls = cli_env
        # When
        result = RUNNER.invoke(cli_module.app, ["build", "--force"])
        # Then
        assert result.exit_code == 0
        assert calls[0][1] is True

    def test_skipped_build_reports_up_to_date(
        self, cli_env: tuple[Path, list[BuildCall]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given an unchanged corpus, then the CLI reports the existing count."""
        # Given
        store = SimpleNamespace(chunk_count=lambda: 11)
        _stub_indexer(monkeypatch, IndexReport(indexed_chunks=0, skipped=True), store)
        # When
        result = RUNNER.invoke(cli_module.app, ["build"])
        # Then
        assert result.exit_code == 0
        assert "Index up to date: 11 chunks in collection test_docs" in result.output

    def test_empty_corpus_reports_unchanged(
        self, cli_env: tuple[Path, list[BuildCall]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given a corpus with no files, then the CLI reports nothing indexed."""
        # Given
        _stub_indexer(monkeypatch, IndexReport(indexed_chunks=0, skipped=False))
        # When
        result = RUNNER.invoke(cli_module.app, ["build"])
        # Then
        assert result.exit_code == 0
        assert "No chunks produced" in result.output


class TestInspectCommand:
    """Tests for the rag-index inspect command."""

    def _invoke(self, monkeypatch: pytest.MonkeyPatch, info: CollectionInfo) -> Any:
        """Invoke inspect with a stubbed config and store."""
        cfg = SimpleNamespace(
            indexer_config=SimpleNamespace(backend=SimpleNamespace(value="qdrant"))
        )
        store = SimpleNamespace(describe=lambda: info)
        monkeypatch.setattr(cli_module, "load_app_config", lambda path: cfg)
        monkeypatch.setattr(cli_module, "get_vector_store", lambda config: store)
        return RUNNER.invoke(cli_module.app, ["inspect"])

    def test_prints_collection_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given an indexed collection, then inspect prints model, dim, and count."""
        # Given
        info = CollectionInfo(
            exists=True,
            collection="fastapi_docs",
            model_id="BAAI/bge-small-en-v1.5",
            dim=384,
            chunk_count=1200,
        )
        # When
        result = self._invoke(monkeypatch, info)
        # Then
        assert result.exit_code == 0
        assert "Collection fastapi_docs (qdrant)" in result.output
        assert "BAAI/bge-small-en-v1.5" in result.output
        assert "384" in result.output
        assert "1200" in result.output

    def test_reports_not_indexed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a missing collection, then inspect reports it as not indexed."""
        # Given
        info = CollectionInfo(
            exists=False,
            collection="fastapi_docs",
            model_id=None,
            dim=None,
            chunk_count=0,
        )
        # When
        result = self._invoke(monkeypatch, info)
        # Then
        assert result.exit_code == 0
        assert "not indexed" in result.output

    def test_reports_unknown_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Given a collection without a meta sentinel, then model/dim are unknown."""
        # Given
        info = CollectionInfo(
            exists=True,
            collection="fastapi_docs",
            model_id=None,
            dim=None,
            chunk_count=0,
        )
        # When
        result = self._invoke(monkeypatch, info)
        # Then
        assert result.exit_code == 0
        assert "unknown" in result.output
