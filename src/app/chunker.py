"""Naive and structural character chunkers over a document directory."""

import ast
import logging
import re
from pathlib import Path

from src import ROOT
from src.schemas.retrieval import Chunk
from src.schemas.types import ChunkStrategyEnum

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 2000

_SETEXT_RE = re.compile(r"^(=+|-+)[ \t]*$")

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "Chunk",
    "chunk_directory",
    "chunk_markdown_text",
    "chunk_python_text",
    "chunk_text",
]


def _validate_chunk_params(chunk_size: int, overlap: int) -> int:
    """Validate sizes, clamp overlap, and return the sliding-window stride."""
    # Validate inputs.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        clamped = chunk_size // 2
        logger.warning(
            "Clamping overlap %s to %s for chunk_size %s", overlap, clamped, chunk_size
        )
        overlap = clamped
    # Compute stride.
    return chunk_size - overlap


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
) -> list[str]:
    """Split text into fixed-size character slices with sliding-window overlap.

    Parameters
    ----------
    text:
        Source text to split.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    overlap:
        Characters reused between consecutive chunks. Must be non-negative.

    Returns
    -------
    list[str]
        Character slices in document order. Empty input returns an empty list.

    Notes
    -----
    - ``overlap >= chunk_size`` is clamped to ``chunk_size // 2`` with a warning log.

    """
    # Validate inputs and compute stride.
    stride = _validate_chunk_params(chunk_size, overlap)
    # Return early for empty input.
    if not text:
        return []
    # Slice with sliding window.
    return [text[start : start + chunk_size] for start in range(0, len(text), stride)]


def _is_atx_header(line: str) -> bool:
    """Check whether a single line is an ATX markdown header.

    Parameters
    ----------
    line:
        One source line without its trailing newline.

    Returns
    -------
    bool
        True for ``#`` through ``######`` followed by space/tab or end of line.

    """
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False
    level = len(stripped) - len(stripped.lstrip("#"))
    if level > 6:
        return False
    rest = stripped[level:]
    return rest == "" or rest[0] in (" ", "\t")


def _split_markdown_sections(text: str) -> list[tuple[int, str]]:
    """Split markdown into header-led sections with source offsets.

    Parameters
    ----------
    text:
        Full markdown document text.

    Returns
    -------
    list[tuple[int, str]]
        ``(start_char, section_text)`` pairs in document order, each starting
        at its header line. Text before the first header is its own section.

    """
    # Split keeping ends so offsets map back to the source exactly.
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    # Record the start line of each section.
    starts: list[int] = [0]
    in_fence = False
    in_frontmatter = lines[0].strip() == "---"
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        # Track fenced code spans, whose ``#`` lines are not headers.
        bare = raw.lstrip()
        if bare.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Skip YAML frontmatter so its ``---`` never reads as Setext.
        if in_frontmatter:
            if 0 < index and stripped == "---":
                in_frontmatter = False
            continue
        # ATX header opens a section at its own line.
        if _is_atx_header(raw.rstrip("\n")):
            if index != starts[-1]:
                starts.append(index)
            continue
        # Setext underline promotes the previous text line to a header.
        if _SETEXT_RE.match(stripped) and index > 0 and lines[index - 1].strip():
            header = index - 1
            if header != starts[-1]:
                starts.append(header)
    # Slice sections as exact source ranges.
    line_offsets: list[int] = [0] * (len(lines) + 1)
    for index, raw in enumerate(lines):
        line_offsets[index + 1] = line_offsets[index] + len(raw)
    sections: list[tuple[int, str]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        piece = "".join(lines[start:end])
        if piece.strip():
            sections.append((line_offsets[start], piece))
    return sections


def _pack_sections_with_stride(
    sections: list[tuple[int, str]],
    chunk_size: int,
    stride: int,
) -> list[tuple[int, str]]:
    """Pack sections using a pre-validated stride.

    Parameters
    ----------
    sections:
        ``(start_char, text)`` units in document order.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    stride:
        Sliding-window step from ``_validate_chunk_params``.

    Returns
    -------
    list[tuple[int, str]]
        ``(start_char, chunk_text)`` pairs in document order.

    """
    # Pack with the validated stride; callers validate once upstream.
    packed: list[tuple[int, str]] = []
    current_start = 0
    current_parts: list[str] = []
    current_len = 0
    for start, piece in sections:
        # Oversized unit falls back to blind slicing on its own.
        if len(piece) > chunk_size:
            if current_parts:
                packed.append((current_start, "".join(current_parts)))
                current_parts = []
                current_len = 0
            for offset in range(0, len(piece), stride):
                sliver = piece[offset : offset + chunk_size]
                packed.append((start + offset, sliver))
            continue
        # Flush when the next whole section would overflow.
        if current_parts and current_len + len(piece) > chunk_size:
            packed.append((current_start, "".join(current_parts)))
            current_parts = []
            current_len = 0
        if not current_parts:
            current_start = start
        current_parts.append(piece)
        current_len += len(piece)
    # Flush the tail.
    if current_parts:
        packed.append((current_start, "".join(current_parts)))
    return packed


def _pack_sections(
    sections: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, str]]:
    """Pack whole sections to size, char-slicing only oversized ones.

    Parameters
    ----------
    sections:
        ``(start_char, text)`` units in document order.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    overlap:
        Characters reused between consecutive fallback slices.

    Returns
    -------
    list[tuple[int, str]]
        ``(start_char, chunk_text)`` pairs in document order.

    """
    # Validate once so fallback slicing shares the clamped stride.
    stride = _validate_chunk_params(chunk_size, overlap)
    return _pack_sections_with_stride(sections, chunk_size, stride)


def chunk_markdown_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
) -> list[str]:
    """Split markdown on headers, packing whole sections to size.

    Parameters
    ----------
    text:
        Source markdown text to split.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    overlap:
        Characters reused between consecutive fallback slices. Applies only
        when a single section exceeds ``chunk_size``.

    Returns
    -------
    list[str]
        Chunks in document order. Empty input returns an empty list.

    """
    if not text:
        return []
    sections = _split_markdown_sections(text)
    if not sections:
        return []
    return [piece for _, piece in _pack_sections(sections, chunk_size, overlap)]


def _split_python_blocks(text: str) -> list[tuple[int, str]]:
    """Split Python source into top-level def/class blocks with offsets.

    Parameters
    ----------
    text:
        Full Python module text.

    Returns
    -------
    list[tuple[int, str]]
        ``(start_char, block_text)`` pairs in document order. The leading
        preamble (imports, docstring) is its own block. Gaps between blocks
        attach to the preceding block so no text is lost.

    Raises
    ------
    SyntaxError
        If the module does not parse; callers fall back to naive slicing.

    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    # Map line numbers to character offsets.
    line_offsets: list[int] = [0] * (len(lines) + 1)
    for index, raw in enumerate(lines):
        line_offsets[index + 1] = line_offsets[index] + len(raw)
    tree = ast.parse(text)
    bounds: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = node.lineno
            for deco in node.decorator_list:
                first = min(first, deco.lineno)
            bounds.append((first, node.end_lineno or first))
    # No def/class blocks: the whole file is one unit.
    if not bounds:
        return [(0, text)] if text.strip() else []
    # Slice blocks, attaching gaps to the preceding block.
    blocks: list[tuple[int, str]] = []
    first_start = bounds[0][0] - 1
    if first_start > 0:
        preamble = "".join(lines[:first_start])
        if preamble.strip():
            blocks.append((0, preamble))
    for pos, (start_line, _end_line) in enumerate(bounds):
        start_idx = start_line - 1
        if pos + 1 < len(bounds):
            end_idx = bounds[pos + 1][0] - 1
        else:
            end_idx = len(lines)
        piece = "".join(lines[start_idx:end_idx])
        if piece.strip():
            blocks.append((line_offsets[start_idx], piece))
    return blocks


def chunk_python_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
) -> list[str]:
    """Split Python source on top-level def/class blocks.

    Parameters
    ----------
    text:
        Source Python text to split.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    overlap:
        Characters reused between consecutive fallback slices. Applies only
        when a single block exceeds ``chunk_size``.

    Returns
    -------
    list[str]
        Chunks in document order. Empty input returns an empty list.
        Unparseable source falls back to naive slicing.

    """
    if not text:
        return []
    try:
        blocks = _split_python_blocks(text)
    except SyntaxError:
        logger.warning("Falling back to naive chunking for unparseable Python")
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not blocks:
        return []
    return [piece for _, piece in _pack_sections(blocks, chunk_size, overlap)]


def chunk_directory(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
    strategy: ChunkStrategyEnum | str = ChunkStrategyEnum.NAIVE,
) -> list[Chunk]:
    """Rglob markdown and python files under path and chunk each file.

    Parameters
    ----------
    path:
        Directory to walk recursively.
    chunk_size:
        Maximum characters per chunk. Must be positive.
    overlap:
        Characters reused between consecutive chunks. Must be non-negative.
        Structural strategies apply it only inside fallback slices.
    strategy:
        ``NAIVE`` for fixed-size slicing or ``STRUCTURAL`` for header-led
        markdown sections and def/class-led Python blocks.

    Returns
    -------
    list[Chunk]
        Chunks in sorted document order, each with repo-relative identity.

    Notes
    -----
    - Discovery is recursive ``path.rglob("*")`` filtered to ``{".md", ".py"}``.
    - Empty or whitespace-only files are skipped.
    - Non-UTF-8 or unreadable files are skipped with a warning log.
    - Symlinked files are deduped by resolved path; ``doc_path`` is the resolved target.
    - A symlink pointing outside the corpus falls back to the link path with a warning.
    - ``doc_path`` is relative to ROOT when inside ROOT, else relative to ``path``.

    """
    # Validate once; reuse the stride downstream without re-validating.
    stride = _validate_chunk_params(chunk_size, overlap)
    # Normalize the strategy so plain strings from config still dispatch.
    resolved_strategy = (
        strategy
        if isinstance(strategy, ChunkStrategyEnum)
        else ChunkStrategyEnum(strategy)
    )
    # Guard the input path.
    if not path.exists():
        raise FileNotFoundError(f"chunk path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"chunk path is not a directory: {path}")
    # Discover candidate files, deduping symlinks by resolved path.
    candidates = sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix in {".md", ".py"}
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        files.append(candidate)
    # Chunk each file.
    chunks: list[Chunk] = []
    root_resolved: Path = ROOT.resolve()
    base_resolved: Path = path.resolve()
    for file in files:
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file %s", file)
            continue
        except OSError:
            logger.warning("Skipping unreadable file %s", file)
            continue
        if not content.strip():
            continue
        # Resolve display path.
        try:
            doc_path = file.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            try:
                doc_path = file.resolve().relative_to(base_resolved).as_posix()
            except ValueError:
                logger.warning(
                    "Symlink target outside corpus for %s, using link path", file
                )
                doc_path = file.relative_to(path).as_posix()
        # Slice and record offsets.
        if resolved_strategy == ChunkStrategyEnum.STRUCTURAL and file.suffix == ".md":
            units = _pack_sections_with_stride(
                _split_markdown_sections(content), chunk_size, stride
            )
        elif resolved_strategy == ChunkStrategyEnum.STRUCTURAL and file.suffix == ".py":
            try:
                units = _pack_sections_with_stride(
                    _split_python_blocks(content), chunk_size, stride
                )
            except SyntaxError:
                logger.warning("Falling back to naive chunking for %s", file)
                slices = [
                    content[start : start + chunk_size]
                    for start in range(0, len(content), stride)
                ]
                units = [(index * stride, piece) for index, piece in enumerate(slices)]
        else:
            slices = [
                content[start : start + chunk_size]
                for start in range(0, len(content), stride)
            ]
            units = [(index * stride, piece) for index, piece in enumerate(slices)]
        for index, (start, piece) in enumerate(units):
            end = start + len(piece)
            token_count = (len(piece) + 3) // 4
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_path}#{index:04d}",
                    chunk_index=index,
                    text=piece,
                    start_char=start,
                    end_char=end,
                    token_count=token_count,
                    doc_path=doc_path,
                )
            )
    return chunks
