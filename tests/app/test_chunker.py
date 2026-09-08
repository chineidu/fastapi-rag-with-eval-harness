"""Tests for the naive character chunker."""

import logging
from pathlib import Path
from typing import Any

import pytest

from src.app.chunker import DEFAULT_CHUNK_SIZE, Chunk, chunk_directory, chunk_text
from src.schemas.retrieval import Chunk as CanonicalChunk


class TestChunkText:
    def test_default_chunk_size_is_2000(self) -> None:
        # Given / When / Then.
        assert DEFAULT_CHUNK_SIZE == 2000

    def test_empty_returns_no_chunks(self) -> None:
        # Given
        text = ""
        # When
        result = chunk_text(text)
        # Then
        assert result == []

    def test_shorter_than_size_returns_one_chunk(self) -> None:
        # Given
        text = "hello"
        # When
        result = chunk_text(text, chunk_size=10)
        # Then
        assert result == ["hello"]

    def test_exact_size_returns_one_chunk(self) -> None:
        # Given
        text = "x" * 10
        # When
        result = chunk_text(text, chunk_size=10)
        # Then
        assert result == [text]

    def test_remainder_last_chunk_is_smaller(self) -> None:
        # Given
        text = "x" * 15
        # When
        result = chunk_text(text, chunk_size=10)
        # Then
        assert len(result) == 2
        assert len(result[0]) == 10
        assert len(result[1]) == 5

    def test_overlap_zero_reconstructs_input(self) -> None:
        # Given
        text = "abcdefghij" * 3  # 30 chars
        # When
        result = chunk_text(text, chunk_size=10, overlap=0)
        # Then
        assert "".join(result) == text
        assert [len(c) for c in result] == [10, 10, 10]

    def test_overlap_slides_by_stride(self) -> None:
        # Given
        text = "0123456789" * 2  # 20 chars, chunk_size=10, overlap=4, stride=6.
        # When
        result = chunk_text(text, chunk_size=10, overlap=4)
        # Then
        assert result[0] == text[0:10]
        assert result[1] == text[6:16]
        assert result[2] == text[12:20]
        assert result[3] == text[18:20]

    def test_max_overlap_uses_stride_one(self) -> None:
        # Given
        text = "abcde"
        # When
        result = chunk_text(text, chunk_size=5, overlap=4)
        # Then
        assert result == ["abcde", "bcde", "cde", "de", "e"]

    @pytest.mark.parametrize("bad", [-1, 0])
    def test_bad_chunk_size_raises(self, bad: int) -> None:
        # Given
        text = "hello"
        # When / Then
        with pytest.raises(ValueError, match="must be positive"):
            chunk_text(text, chunk_size=bad)

    def test_negative_overlap_raises(self) -> None:
        # Given
        text = "hello"
        # When / Then
        with pytest.raises(ValueError, match="non-negative"):
            chunk_text(text, chunk_size=10, overlap=-1)

    def test_overlap_clamped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given
        text = "x" * 25
        # When
        with caplog.at_level(logging.WARNING):
            result = chunk_text(text, chunk_size=10, overlap=10)
        explicit = chunk_text(text, chunk_size=10, overlap=5)
        # Then
        assert result == explicit
        assert "Clamping overlap" in caplog.text


class TestChunkType:
    def test_chunk_is_canonical_schema_type(self) -> None:
        # Given / When / Then.
        assert Chunk is CanonicalChunk

    def test_chunk_requires_keyword_args(self) -> None:
        # Given
        kwargs = {
            "chunk_id": "a.md#0000",
            "chunk_index": 0,
            "text": "hi",
            "start_char": 0,
            "end_char": 2,
            "token_count": 1,
            "doc_path": "a.md",
        }
        # When
        chunk = Chunk(**kwargs)
        # Then
        assert chunk.doc_path == "a.md"
        assert chunk.token_count == 1
        untyped: Any = Chunk
        with pytest.raises(TypeError):
            untyped("a.md#0000", 0, "hi", 0, 2, 1, "a.md")


class TestChunkDirectory:
    def test_discovers_md_and_py_sorted_ignores_others(self, tmp_path: Path) -> None:
        # Given
        (tmp_path / "z.md").write_text("zzzz")
        (tmp_path / "a.md").write_text("aaaa")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.py").write_text("print(1)\n")
        (tmp_path / "c.txt").write_text("ignored")
        # When
        result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["a.md", "sub/b.py", "z.md"]

    def test_skips_empty_and_whitespace(self, tmp_path: Path) -> None:
        # Given
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "ws.md").write_text("  \n\t  \n")
        (tmp_path / "real.md").write_text("hello")
        # When
        result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["real.md"]

    def test_offsets_ids_and_tokens(self, tmp_path: Path) -> None:
        # Given
        text = "abcdefghij" * 3  # 30 chars
        (tmp_path / "foo.md").write_text(text)
        # When
        result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert len(result) == 3
        for i, chunk in enumerate(result):
            assert isinstance(chunk, Chunk)
            assert chunk.doc_path == "foo.md"
            assert chunk.chunk_index == i
            assert chunk.chunk_id == f"foo.md#{i:04d}"
            assert chunk.start_char == i * 10
            assert chunk.end_char == i * 10 + len(chunk.text)
            assert text[chunk.start_char : chunk.end_char] == chunk.text
            assert chunk.token_count == (len(chunk.text) + 3) // 4

    def test_doc_path_input_relative_outside_root(self, tmp_path: Path) -> None:
        # Given
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "inner.md").write_text("hello")
        # When
        result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["sub/inner.md"]

    def test_doc_path_root_relative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setattr("src.app.chunker.ROOT", tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("hello")
        # When
        result = chunk_directory(corpus, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["corpus/a.md"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        # Given
        missing = tmp_path / "nope"
        # When / Then
        with pytest.raises(FileNotFoundError, match="does not exist"):
            chunk_directory(missing)

    def test_file_path_raises(self, tmp_path: Path) -> None:
        # Given
        file_path = tmp_path / "a.md"
        file_path.write_text("hello")
        # When / Then
        with pytest.raises(NotADirectoryError, match="not a directory"):
            chunk_directory(file_path)

    def test_skips_non_utf8_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given
        (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00bad")
        (tmp_path / "good.md").write_text("hello")
        # When
        with caplog.at_level(logging.WARNING):
            result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["good.md"]
        assert "non-UTF-8" in caplog.text

    def test_symlink_not_duplicated(self, tmp_path: Path) -> None:
        # Given
        real = tmp_path / "real.md"
        real.write_text("hello")
        link = tmp_path / "link.md"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("symlinks not supported")
        # When
        result = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["real.md"]
        assert len(result) == 1

    def test_external_symlink_uses_link_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given
        (tmp_path / "outside.md").write_text("hello")
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        try:
            (corpus / "ext.md").symlink_to(tmp_path / "outside.md")
        except OSError:
            pytest.skip("symlinks not supported")
        # When
        with caplog.at_level(logging.WARNING):
            result = chunk_directory(corpus, chunk_size=10)
        # Then
        assert [c.doc_path for c in result] == ["ext.md"]
        assert "outside corpus" in caplog.text

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        # Given
        (tmp_path / "a.md").write_text("hello\n" * 20)
        (tmp_path / "b.md").write_text("world\n" * 20)
        # When
        first = chunk_directory(tmp_path, chunk_size=10)
        second = chunk_directory(tmp_path, chunk_size=10)
        # Then
        assert [(c.chunk_id, c.text) for c in first] == [
            (c.chunk_id, c.text) for c in second
        ]

    def test_overlap_clamp_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Given
        (tmp_path / "a.md").write_text("x" * 25)
        # When
        with caplog.at_level(logging.WARNING):
            result = chunk_directory(tmp_path, chunk_size=10, overlap=20)
        explicit = chunk_directory(tmp_path, chunk_size=10, overlap=5)
        # Then
        assert [(c.chunk_id, c.text) for c in result] == [
            (c.chunk_id, c.text) for c in explicit
        ]
        assert "Clamping overlap" in caplog.text
