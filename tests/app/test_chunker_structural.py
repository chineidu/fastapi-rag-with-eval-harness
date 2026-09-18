"""Tests for the structural markdown and Python chunkers."""

from pathlib import Path

from src.app.chunker import (
    chunk_directory,
    chunk_markdown_text,
    chunk_python_text,
    chunk_text,
)
from src.schemas.types import ChunkStrategyEnum


class TestChunkMarkdownText:
    def test_empty_returns_no_chunks(self) -> None:
        # Given
        text = ""
        # When
        result = chunk_markdown_text(text)
        # Then
        assert result == []

    def test_atx_all_levels_split(self) -> None:
        # Given
        text = "# H1\na\n## H2\nb\n### H3\nc\n"
        # When
        result = chunk_markdown_text(text, chunk_size=10)
        # Then
        assert len(result) == 3
        assert result[0].startswith("# H1")
        assert result[1].startswith("## H2")
        assert result[2].startswith("### H3")

    def test_setext_underline_splits(self) -> None:
        # Given
        text = "Intro\nbody\nMy Section\n---\nmore\n"
        # When
        result = chunk_markdown_text(text, chunk_size=20)
        # Then
        assert len(result) == 2
        assert "My Section" in result[1]

    def test_fenced_headers_ignored(self) -> None:
        # Given
        text = "# Real\nbody\n```\n# Not a header\ncode\n```\n## Next\nmore\n"
        # When
        result = chunk_markdown_text(text, chunk_size=40)
        # Then
        assert len(result) == 2
        assert "# Not a header" in result[0]
        assert result[1].startswith("## Next")

    def test_frontmatter_not_a_section(self) -> None:
        # Given
        text = "---\ntitle: x\n---\n# H\nbody\n"
        # When
        result = chunk_markdown_text(text, chunk_size=20)
        # Then
        assert len(result) == 2
        assert result[1].startswith("# H")

    def test_oversized_section_char_sliced(self) -> None:
        # Given
        text = "x" * 100
        # When
        result = chunk_markdown_text(text, chunk_size=30)
        # Then
        assert result == chunk_text(text, chunk_size=30)
        assert "".join(result) == text

    def test_packed_sections_reconstruct(self) -> None:
        # Given
        text = "# A\none\n## B\ntwo\n"
        # When
        result = chunk_markdown_text(text, chunk_size=1000)
        # Then
        assert "".join(result) == text

    def test_nested_headers_flattened(self) -> None:
        # Given
        text = "# A\ncontent a\n## B\ncontent b\n"
        # When
        result = chunk_markdown_text(text, chunk_size=15)
        # Then
        assert len(result) == 2
        assert result[0].startswith("# A")
        assert result[1].startswith("## B")

    def test_trailing_blank_lines_stay_with_section(self) -> None:
        # Given
        text = "# H\n\n\n"
        # When
        result = chunk_markdown_text(text, chunk_size=1000)
        # Then
        assert len(result) == 1
        assert result[0].startswith("# H")
        assert "".join(result) == text

    def test_whitespace_only_returns_no_chunks(self) -> None:
        # Given
        text = "   \n\n  \n"
        # When
        result = chunk_markdown_text(text, chunk_size=1000)
        # Then
        assert result == []


class TestChunkPythonText:
    def test_empty_returns_no_chunks(self) -> None:
        # Given
        text = ""
        # When
        result = chunk_python_text(text)
        # Then
        assert result == []

    def test_top_level_blocks_split(self) -> None:
        # Given
        text = "import os\n\n\ndef foo():\n    return 1\n\n\ndef bar():\n    return 2\n"
        # When
        result = chunk_python_text(text, chunk_size=30)
        # Then
        assert len(result) == 3
        assert "import os" in result[0]
        assert "def foo" in result[1]
        assert "def bar" in result[2]

    def test_decorator_starts_block(self) -> None:
        # Given
        text = "@app.get()\ndef foo():\n    return 1\n"
        # When
        result = chunk_python_text(text, chunk_size=1000)
        # Then
        assert len(result) == 1
        assert result[0].startswith("@app.get()")

    def test_syntax_error_falls_back(self) -> None:
        # Given
        text = "def broken(:\n  pass\n"
        # When
        result = chunk_python_text(text, chunk_size=10)
        # Then
        assert result == chunk_text(text, chunk_size=10)

    def test_oversized_block_char_sliced(self) -> None:
        # Given
        text = "def foo():\n" + "    x = 1\n" * 100
        # When
        result = chunk_python_text(text, chunk_size=50)
        # Then
        assert result == chunk_text(text, chunk_size=50)

    def test_no_blocks_returns_whole(self) -> None:
        # Given
        text = "x = 1\ny = 2\n"
        # When
        result = chunk_python_text(text, chunk_size=1000)
        # Then
        assert result == [text]


class TestChunkDirectoryStructural:
    def test_structural_offsets_match_source(self, tmp_path: Path) -> None:
        # Given
        text = "# A\none\n## B\ntwo\n"
        (tmp_path / "doc.md").write_text(text)
        # When
        result = chunk_directory(
            tmp_path, chunk_size=10, strategy=ChunkStrategyEnum.STRUCTURAL
        )
        # Then
        for chunk in result:
            assert text[chunk.start_char : chunk.end_char] == chunk.text

    def test_python_offsets_match_source(self, tmp_path: Path) -> None:
        # Given
        text = "import os\n\n\ndef foo():\n    return 1\n"
        (tmp_path / "mod.py").write_text(text)
        # When
        result = chunk_directory(
            tmp_path, chunk_size=20, strategy=ChunkStrategyEnum.STRUCTURAL
        )
        # Then
        assert result
        for chunk in result:
            assert text[chunk.start_char : chunk.end_char] == chunk.text

    def test_naive_default_matches_explicit(self, tmp_path: Path) -> None:
        # Given
        (tmp_path / "a.md").write_text("# H\nbody text here\n")
        # When
        default = chunk_directory(tmp_path, chunk_size=10)
        explicit = chunk_directory(
            tmp_path, chunk_size=10, strategy=ChunkStrategyEnum.NAIVE
        )
        # Then
        assert [(c.chunk_id, c.text) for c in default] == [
            (c.chunk_id, c.text) for c in explicit
        ]

    def test_string_strategy_coerced(self, tmp_path: Path) -> None:
        # Given
        (tmp_path / "a.md").write_text("# H\nbody\n")
        # When
        result = chunk_directory(tmp_path, chunk_size=10, strategy="structural")
        # Then
        assert [c.doc_path for c in result] == ["a.md"]
