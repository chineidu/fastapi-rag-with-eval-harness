"""Tests for the rag-index CLI."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from src.app import cli as cli_module

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

        def build(self, corpus_roots: list[Path], *, force: bool = False) -> int:
            calls.append((list(corpus_roots), force))
            return 7

    monkeypatch.setattr(cli_module, "Indexer", RecordingIndexer)
    return tmp_path, calls


class TestBuildCommand:
    """Tests for the single-command rag-index build surface."""

    def test_defaults_to_both_corpus_roots(
        self, cli_env: tuple[Path, list[BuildCall]]
    ) -> None:
        """Given no --corpus, then both default corpus roots are indexed."""
        # Given
        root, calls = cli_env
        # When
        result = RUNNER.invoke(cli_module.app, [])
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
            ["--corpus", "docs/fastapi/docs_src", "--corpus", "extra"],
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
        result = RUNNER.invoke(cli_module.app, ["--corpus", "does/not/exist"])
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
        result = RUNNER.invoke(cli_module.app, ["--corpus", ""])
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
        result = RUNNER.invoke(cli_module.app, ["--force"])
        # Then
        assert result.exit_code == 0
        assert calls[0][1] is True
