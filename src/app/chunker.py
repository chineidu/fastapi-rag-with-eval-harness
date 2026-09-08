"""Naive fixed-size character chunker over a document directory."""

import logging
from pathlib import Path

from src import ROOT
from src.schemas.retrieval import Chunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 2000

__all__ = ["DEFAULT_CHUNK_SIZE", "Chunk", "chunk_directory", "chunk_text"]


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


def chunk_directory(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = 0,
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
    # Validate inputs and compute stride shared with chunk_text.
    stride = _validate_chunk_params(chunk_size, overlap)
    clamped_overlap = chunk_size - stride
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
        slices = chunk_text(content, chunk_size=chunk_size, overlap=clamped_overlap)
        for index, piece in enumerate(slices):
            start = index * stride
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
