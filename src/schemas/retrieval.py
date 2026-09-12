"""Runtime value types for the retrieval pipeline (chunker, indexer, retriever).

Parallel to :mod:`src.schemas.harness` for the eval harness's runtime types.
"""

from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class Chunk:
    """One chunk of a corpus document.

    Parameters
    ----------
    chunk_id : str
        Stable identifier of the form ``"{doc_path}#{chunk_index:04d}"``.
        Deterministic so the same corpus produces the same IDs across runs.
    chunk_index : int
        Zero-based index of this chunk within its source document.
    text : str
        Raw chunk text as it will be passed to the embedder.
    start_char : int
        Inclusive offset of this chunk within the source document's text.
    end_char : int
        Exclusive offset of this chunk within the source document's text.
    token_count : int
        Naive estimate as ``ceil(len(text) / 4)``, minimum 1 when text is non-empty.
    doc_path : str
        Relative path to the source document (matches the ground truth
        format: ``docs/en/docs/...md`` or ``docs/fastapi/docs_src/...py``).
        ROOT-relative when inside ROOT, else relative to the input path.

    """

    chunk_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    doc_path: str


@dataclass(slots=True, frozen=True)
class SearchHit:
    """One retrieval result with its stored chunk metadata."""

    chunk_id: str
    doc_path: str
    chunk_index: int
    text: str
    score: float


@dataclass(slots=True, frozen=True)
class CollectionInfo:
    """Read-only description of one vector store collection.

    Parameters
    ----------
    exists : bool
        Whether the collection exists in the backend.
    collection : str
        Collection name.
    model_id : str | None
        Embedding model recorded at index time, or ``None`` when absent.
    dim : int | None
        Vector dimension recorded at index time, or ``None`` when absent.
    chunk_count : int
        Number of indexed chunks, excluding the meta sentinel.

    """

    exists: bool
    collection: str
    model_id: str | None
    dim: int | None
    chunk_count: int


@dataclass(slots=True, frozen=True)
class IndexReport:
    """Outcome of one indexer build call.

    Parameters
    ----------
    indexed_chunks : int
        Chunks embedded and upserted during this call.
    skipped : bool
        ``True`` when the corpus fingerprint matched and the index was
        left untouched.

    """

    indexed_chunks: int
    skipped: bool


__all__ = ["Chunk", "CollectionInfo", "IndexReport", "SearchHit"]
