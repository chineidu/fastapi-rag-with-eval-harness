"""Persisted tantivy BM25 index plus RRF fusion with dense hits."""

import importlib
import logging
import re
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from src.schemas.retrieval import Chunk, SearchHit

logger = logging.getLogger(__name__)

__all__ = ["TantivyIndex", "rrf_fuse"]

# Matches the default tantivy tokenizer: lowercase alphanumeric runs.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_MIN_TOKEN_LEN = 3

_STOPWORDS_FILE = files("src.app").joinpath("stopwords_en.txt")


def _load_stopwords() -> frozenset[str]:
    """Load the canonical Snowball English stopword list from package data.

    Returns
    -------
    frozenset[str]
        Stopwords, one per line, skipping blank lines and ``#`` comments.

    """
    text = _STOPWORDS_FILE.read_text(encoding="utf-8")
    return frozenset(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


_STOPWORDS = _load_stopwords()


def _query_tokens(text: str) -> list[str]:
    """Tokenize query text the way the default tantivy tokenizer does.

    Parameters
    ----------
    text:
        Raw query text.

    Returns
    -------
    list[str]
        Lowercase alphanumeric tokens in first-seen order, deduplicated,
        with stopwords and tokens shorter than ``_MIN_TOKEN_LEN`` removed.

    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return list(
        dict.fromkeys(
            token
            for token in tokens
            if len(token) >= _MIN_TOKEN_LEN and token not in _STOPWORDS
        )
    )


def _require_tantivy() -> Any:
    """Import tantivy lazily so this module loads without the optional dep.

    Returns
    -------
    Any
        The imported ``tantivy`` module.

    Raises
    ------
    ImportError
        If ``tantivy`` is not installed.

    """
    try:
        return importlib.import_module("tantivy")
    except ImportError as exc:
        raise ImportError(
            "tantivy is not installed; run `uv add tantivy` to enable the lexical index"
        ) from exc


def _build_schema(tantivy: Any) -> Any:
    """Build the tantivy schema shared by index builds and searches.

    Parameters
    ----------
    tantivy:
        The imported ``tantivy`` module.

    Returns
    -------
    Any
        Schema with stored ``chunk_id``/``doc_path``/``chunk_index`` and
        indexed ``text``. All fields are text so the binding surface stays
        to one field type.

    """
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True)
    builder.add_text_field("doc_path", stored=True)
    builder.add_text_field("chunk_index", stored=True)
    builder.add_text_field("text", stored=True)
    return builder.build()


def _stored_field(doc: Any, name: str, default: str = "") -> str:
    """Read one stored field from a tantivy document.

    Parameters
    ----------
    doc:
        Document returned by ``searcher.doc``.
    name:
        Stored field name.
    default:
        Fallback when the field is absent or empty.

    Returns
    -------
    str
        The first stored value, or ``default``.

    """
    # Missing stored fields mean a foreign or partial document; fall back
    # instead of failing the whole query.
    try:
        values = doc[name]
    except KeyError, TypeError:
        return default
    if isinstance(values, (list, tuple)):
        return str(values[0]) if values else default
    return str(values)


class TantivyIndex:
    """Persisted BM25 lexical index over chunk texts.

    Built once at ``rag-index build`` time and loaded at query time.
    Chunk identity follows :class:`Chunk` so dense and sparse hits fuse
    on ``chunk_id``.
    """

    def __init__(self, index_dir: Path) -> None:
        """Bind the index to its on-disk directory.

        Parameters
        ----------
        index_dir:
            Directory holding the persisted tantivy index.

        """
        self._dir = index_dir

    def build(self, chunks: list[Chunk]) -> None:
        """Build or rebuild the lexical index from chunks.

        Parameters
        ----------
        chunks:
            Chunks to index with ``doc_path`` stored and ``text`` indexed.

        """
        # Clear stale state so an empty rebuild cannot leave old hits behind.
        if not chunks:
            if self._dir.exists():
                shutil.rmtree(self._dir)
            logger.warning("No chunks for lexical index at %s", self._dir)
            return
        tantivy = _require_tantivy()
        # Rebuild from scratch so stale documents cannot linger.
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        index = tantivy.Index(_build_schema(tantivy), path=str(self._dir))
        writer = index.writer()
        for chunk in chunks:
            doc = tantivy.Document()
            doc.add_text("chunk_id", chunk.chunk_id)
            doc.add_text("doc_path", chunk.doc_path)
            doc.add_text("chunk_index", str(chunk.chunk_index))
            doc.add_text("text", chunk.text)
            writer.add_document(doc)
        writer.commit()
        logger.info("Built lexical index with %d chunks at %s", len(chunks), self._dir)

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the top-k lexical hits for a query.

        Parameters
        ----------
        query:
            Raw user query text.
        k:
            Maximum hits to return. Must be positive.

        Returns
        -------
        list[SearchHit]
            Ranked chunk hits in descending score order. Empty when the
            query is blank or the index was never built.

        Raises
        ------
        ValueError
            If ``k`` is not positive.

        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # Blank queries and missing indexes yield no sparse hits.
        if not query.strip():
            return []
        if not self._dir.exists() or not any(self._dir.iterdir()):
            logger.debug("Lexical index missing at %s", self._dir)
            return []
        tantivy = _require_tantivy()
        schema = _build_schema(tantivy)
        index = tantivy.Index(schema, path=str(self._dir))
        searcher = index.searcher()
        # Build an OR query over tokenized terms; the query parser would
        # reject raw question text containing code and punctuation.
        tokens = _query_tokens(query)
        if not tokens:
            return []
        term_queries = [
            (tantivy.Occur.Should, tantivy.Query.term_query(schema, "text", token))
            for token in tokens
        ]
        results = searcher.search(tantivy.Query.boolean_query(term_queries), k)
        hits: list[SearchHit] = []
        for score, address in results.hits:
            doc = searcher.doc(address)
            chunk_id = _stored_field(doc, "chunk_id")
            doc_path = _stored_field(doc, "doc_path")
            if not chunk_id or not doc_path:
                continue
            try:
                chunk_index = int(_stored_field(doc, "chunk_index", "0"))
            except ValueError:
                chunk_index = 0
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    doc_path=doc_path,
                    chunk_index=chunk_index,
                    text=_stored_field(doc, "text"),
                    score=float(score),
                )
            )
        return hits


def rrf_fuse(
    dense: list[SearchHit],
    sparse: list[SearchHit],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[SearchHit]:
    """Fuse dense and sparse chunk hits with reciprocal rank fusion.

    Parameters
    ----------
    dense:
        Dense vector hits in rank order.
    sparse:
        Sparse BM25 hits in rank order.
    rrf_k:
        RRF smoothing constant. Must be positive.
    dense_weight:
        Multiplier on dense rank contributions. Must be positive.
        Weights are normalized to sum to 1 before scoring.
    sparse_weight:
        Multiplier on sparse rank contributions. Must be positive.
        Weights are normalized to sum to 1 before scoring.

    Returns
    -------
    list[SearchHit]
        Fused hits in descending fused-score order.

    Raises
    ------
    ValueError
        If ``rrf_k``, ``dense_weight``, or ``sparse_weight`` is not positive.

    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    if dense_weight <= 0:
        raise ValueError(f"dense_weight must be positive, got {dense_weight}")
    if sparse_weight <= 0:
        raise ValueError(f"sparse_weight must be positive, got {sparse_weight}")
    # Normalize weights to a ratio summing to 1; scaling is rank-invariant.
    total = dense_weight + sparse_weight
    dense_weight /= total
    sparse_weight /= total
    # Accumulate weighted reciprocal-rank scores keyed by chunk identity.
    fused_scores: dict[str, float] = {}  # {chunk_id: score}
    rep_hits: dict[str, SearchHit] = {}  # {chunk_id: hit}
    for hits, weight in ((dense, dense_weight), (sparse, sparse_weight)):
        for rank, hit in enumerate(hits, start=1):
            fused_scores[hit.chunk_id] = fused_scores.get(
                hit.chunk_id, 0.0
            ) + weight / (rrf_k + rank)
            # Every occurrence scores above; only the first is kept for metadata.
            rep_hits.setdefault(hit.chunk_id, hit)

    # Rank chunk_ids by fused score and attach first-seen metadata to each.
    ranked: list[tuple[str, float]] = sorted(
        fused_scores.items(), key=lambda item: item[1], reverse=True
    )
    return [
        SearchHit(
            chunk_id=rep_hits[chunk_id].chunk_id,
            doc_path=rep_hits[chunk_id].doc_path,
            chunk_index=rep_hits[chunk_id].chunk_index,
            text=rep_hits[chunk_id].text,
            score=score,
        )
        for chunk_id, score in ranked
    ]
