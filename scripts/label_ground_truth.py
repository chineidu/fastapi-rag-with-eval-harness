"""Label ground truth for the eval harness.

Loads queries from ``data/eval_dataset.jsonl``, embeds the corpus, performs
semantic search to find top-30 candidate documents per query, then uses an
LLM judge to determine which candidates are relevant.  Results are written
to ``data/ground_truth.json``.

Usage:
    uv run python -m scripts.label_ground_truth label
    uv run python -m scripts.label_ground_truth label --top-k 30
    uv run python -m scripts.label_ground_truth label --dry-run
"""

import asyncio
import json
import time
from pathlib import Path

import instructor
import numpy as np
import openai
import typer

from src import create_logger
from src.config import app_config, app_settings
from src.embeddings import AbstractEmbedder, get_embedder
from src.prompts import JUDGE_SYSTEM_PROMPT
from src.schemas.ground_truth import (
    CorpusDocument,
    GroundTruthRecord,
    JudgeResponse,
    LabelVerdict,
)
from src.schemas.output import UnifiedEvalRecordSchema
from src.schemas.types import ClassificationLabel, VerdictEnum
from src.utils import read_jsonl
from src.utils.text import clean_query_text

logger = create_logger(name=__name__)

app = typer.Typer(
    help="Label ground truth for the eval harness.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Config shortcuts
# ---------------------------------------------------------------------------

_labeling = app_config.eval_pipeline_config.labeling
_EVAL_DATASET_PATH = Path(app_config.eval_pipeline_config.defaults.eval_dataset_path)
_EMBEDDINGS_CACHE_DIR = Path(app_config.embeddings_config.cache_dir)

# ---------------------------------------------------------------------------
# LLM judge client
# ---------------------------------------------------------------------------

_openai_client = openai.AsyncOpenAI(
    base_url=app_settings.OPENROUTER_BASE_URL,
    api_key=app_settings.OPENROUTER_API_KEY.get_secret_value(),
    timeout=app_config.eval_pipeline_config.classifier.timeout_seconds,
    max_retries=app_config.eval_pipeline_config.classifier.max_retries,
)
_aclient = instructor.from_openai(_openai_client)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus(corpus_md_root: Path, corpus_py_root: Path) -> list[CorpusDocument]:
    """Load all corpus files (.md and .py) as CorpusDocument objects.

    Parameters
    ----------
    corpus_md_root : Path
        Root directory for markdown documentation files.
    corpus_py_root : Path
        Root directory for Python example files.

    Returns
    -------
    list[CorpusDocument]
        Loaded documents with relative paths and content.

    """
    docs: list[CorpusDocument] = []

    # Markdown files
    if corpus_md_root.exists():
        for md_path in sorted(corpus_md_root.rglob("*.md")):
            rel = str(md_path.relative_to(corpus_md_root.parent.parent.parent))
            content = md_path.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                docs.append(CorpusDocument(path=rel, content=content))

    # Python files
    if corpus_py_root.exists():
        for py_path in sorted(corpus_py_root.rglob("*.py")):
            rel = str(py_path.relative_to(corpus_py_root.parent.parent.parent))
            content = py_path.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                docs.append(CorpusDocument(path=rel, content=content))

    logger.info(
        "Loaded %d corpus documents (%s, %s)", len(docs), corpus_md_root, corpus_py_root
    )
    return docs


# ---------------------------------------------------------------------------
# Embedding with .np cache
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, model_id: str) -> tuple[Path, Path]:
    """Return (vectors.npy, paths.json) cache file paths for the given model."""
    safe_model = model_id.replace("/", "_")
    return (
        cache_dir / f"{safe_model}_vectors.npy",
        cache_dir / f"{safe_model}_paths.json",
    )


def _load_or_embed_corpus(
    docs: list[CorpusDocument],
    embedder: AbstractEmbedder,
    cache_dir: Path,
) -> np.ndarray:
    """Embed corpus documents, using cache if available.

    Parameters
    ----------
    docs : list[CorpusDocument
        Loaded corpus documents.
    embedder : AbstractEmbedder
        Embedder instance.
    cache_dir : Path
        Directory for .np cache files.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (n_docs, dim).

    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_path, paths_path = _cache_paths(cache_dir, embedder.model_id)

    if vectors_path.exists() and paths_path.exists():
        cached_paths = json.loads(paths_path.read_text(encoding="utf-8"))
        current_paths: list[str] = [d.path for d in docs]
        if cached_paths == current_paths:
            logger.info("Loading cached embeddings from %s", vectors_path)
            return np.load(vectors_path)

    logger.info("Embedding %d corpus documents...", len(docs))
    start: float = time.monotonic()
    texts: list[str] = [
        f"{d.path}\n\n{d.content[: _labeling.max_content_length]}" for d in docs
    ]
    vectors: list[list[float]] = embedder.embed_texts(texts)
    elapsed: float = time.monotonic() - start
    logger.info("Embedded %d docs in %.1fs", len(docs), elapsed)

    arr = np.array(vectors, dtype=np.float32)
    np.save(vectors_path, arr)
    paths_path.write_text(json.dumps([d.path for d in docs]), encoding="utf-8")
    logger.info("Cached embeddings to %s", vectors_path)

    return arr


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def _cosine_top_k(
    query_vec: list[float],
    corpus_vectors: np.ndarray,
    k: int,
) -> list[int]:
    """Return indices of top-k documents by cosine similarity."""
    n: int = len(corpus_vectors)
    k: int = min(k, n)

    # L2-normalize query and corpus vectors
    q = np.array(query_vec, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    norms = np.linalg.norm(corpus_vectors, axis=1) + 1e-10
    corpus_normed = corpus_vectors / norms[:, np.newaxis]

    # Cosine similarity = dot product of normalized vectors
    sims = corpus_normed @ q_norm

    # Partial sort for efficiency, then sort top-k descending
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    return top_indices.tolist()


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


async def _ajudge_candidate(
    query_text: str,
    answer_text: str | None,
    doc: CorpusDocument,
) -> LabelVerdict:
    """Judge whether a single candidate document is relevant."""
    doc_content = doc.content[: _labeling.max_judge_content_length]
    user_msg = (
        f"Question:\n{query_text}\n\n"
        f"Reference answer:\n{answer_text or '(none)'}\n\n"
        f"Candidate document ({doc.path}):\n{doc_content}"
    )

    try:
        response = await _aclient.completions.create(
            model=app_config.eval_pipeline_config.classifier.model_id,
            response_model=JudgeResponse,
            temperature=app_config.eval_pipeline_config.classifier.temperature,
            seed=app_config.eval_pipeline_config.classifier.seed,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return LabelVerdict(
            verdict=response.verdict,
            rationale=response.rationale,
            confidence=response.confidence,
        )
    except Exception:
        logger.exception("LLM judge failed for doc %s", doc.path)
        return LabelVerdict(
            verdict="irrelevant", rationale="judge failed", confidence=0.0
        )


async def _ajudge_candidates(
    query_text: str,
    answer_text: str | None,
    candidates: list[CorpusDocument],
    concurrency: int,
) -> list[LabelVerdict]:
    """Judge all candidates for a query with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(doc: CorpusDocument) -> LabelVerdict:
        async with semaphore:
            return await _ajudge_candidate(query_text, answer_text, doc)

    return await asyncio.gather(*[_bounded(doc) for doc in candidates])


# ---------------------------------------------------------------------------
# Labeling pipeline
# ---------------------------------------------------------------------------


async def _label_query(
    query: UnifiedEvalRecordSchema,
    corpus_docs: list[CorpusDocument],
    corpus_vectors: np.ndarray,
    embedder: AbstractEmbedder,
    top_k: int,
    concurrency: int,
) -> GroundTruthRecord | None:
    """Label a single query: embed, search, judge, return ground truth."""
    query_text: str = clean_query_text(query.body, query.source, query.title)

    # Embed query
    query_vec: list[float] = (await embedder.aembed_texts([query_text]))[0]

    # Semantic search → top-k candidates
    indices: list[int] = _cosine_top_k(query_vec, corpus_vectors, top_k)
    candidates: list[CorpusDocument] = [corpus_docs[i] for i in indices]

    # LLM judge
    verdicts: list[LabelVerdict] = await _ajudge_candidates(
        query_text, query.answer_text, candidates, concurrency
    )

    # Collect relevant docs
    relevant_docs = [
        doc.path
        for doc, verdict in zip(candidates, verdicts, strict=True)
        if verdict.verdict == VerdictEnum.RELEVANT
    ]

    # If no docs passed the judge, keep the top-1 candidate (avoid empty ground truth)
    if not relevant_docs and candidates:
        relevant_docs = [candidates[0].path]
        logger.warning(
            "No relevant docs found for query %s, keeping top-1 candidate %s",
            query.id,
            candidates[0].path,
        )

    return GroundTruthRecord(
        query_id=query.id,
        label=query.label.value if query.label else ClassificationLabel.UNKNOWN,
        query_text=query_text,
        relevant_docs=relevant_docs,
        source=query.source,
        title=query.title,
        answer_text=query.answer_text,
        url=query.url,
    )


async def _alabel_all(
    queries: list[UnifiedEvalRecordSchema],
    corpus_docs: list[CorpusDocument],
    corpus_vectors: np.ndarray,
    embedder: AbstractEmbedder,
    top_k: int,
    concurrency: int,
    dry_run: bool,
) -> list[GroundTruthRecord]:
    """Label all queries and return ground truth records."""
    results: list[GroundTruthRecord] = []
    total = len(queries)

    for i, query in enumerate(queries, 1):
        logger.info("[%d/%d] Labeling query %s...", i, total, query.id)
        start = time.monotonic()

        if dry_run:
            query_text = clean_query_text(query.body, query.source, query.title)
            record = GroundTruthRecord(
                query_id=query.id,
                label=query.label.value if query.label else ClassificationLabel.UNKNOWN,
                query_text=query_text,
                relevant_docs=["(dry-run)"],
                source=query.source,
                title=query.title,
                answer_text=query.answer_text,
                url=query.url,
            )
        else:
            record = await _label_query(
                query, corpus_docs, corpus_vectors, embedder, top_k, concurrency
            )

        if record is not None:
            results.append(record)
            elapsed = time.monotonic() - start
            logger.info(
                "[%d/%d] %s -> %d relevant docs (%.1fs)",
                i,
                total,
                query.id,
                len(record.relevant_docs),
                elapsed,
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Log the invoked command before running it."""
    logger.info("running label_ground_truth command: %s", ctx.invoked_subcommand)


@app.command()
def label(
    top_k: int = typer.Option(
        _labeling.top_k, help="Candidates per query for LLM judge."
    ),
    concurrency: int = typer.Option(
        _labeling.concurrency, help="Max parallel LLM judge calls."
    ),
    dry_run: bool = typer.Option(False, help="Skip LLM judge, use placeholder docs."),
    output: str = typer.Option(_labeling.ground_truth_output, help="Output JSON path."),
    eval_dataset: str = typer.Option(
        str(_EVAL_DATASET_PATH), help="Input eval dataset JSONL."
    ),
    cache_dir: str = typer.Option(
        str(_EMBEDDINGS_CACHE_DIR), help="Embedding cache directory."
    ),
) -> None:
    """Label ground truth: embed corpus, semantic search, LLM judge."""
    eval_path = Path(eval_dataset)
    out_path = Path(output)
    emb_cache = Path(cache_dir)

    if not eval_path.exists():
        logger.error("Eval dataset not found: %s", eval_path)
        raise typer.Exit(code=1)

    # 1. Load queries
    queries: list[UnifiedEvalRecordSchema] = read_jsonl(
        eval_path, UnifiedEvalRecordSchema
    )
    logger.info("Loaded %d queries from %s", len(queries), eval_path)

    # 2. Load corpus
    corpus_docs: list[CorpusDocument] = _load_corpus(
        Path(_labeling.corpus_md_root), Path(_labeling.corpus_py_root)
    )
    if not corpus_docs:
        logger.error("No corpus documents found")
        raise typer.Exit(code=1)

    # 3. Embed corpus
    embedder = get_embedder(app_config.embeddings_config)
    corpus_vectors = _load_or_embed_corpus(corpus_docs, embedder, emb_cache)
    logger.info("Corpus embeddings shape: %s", corpus_vectors.shape)

    # 4. Label all queries
    start: float = time.monotonic()
    results: list[GroundTruthRecord] = asyncio.run(
        _alabel_all(
            queries, corpus_docs, corpus_vectors, embedder, top_k, concurrency, dry_run
        )
    )
    elapsed: float = time.monotonic() - start

    # 5. Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    categories: dict[str, int] = {}  # {label: count}
    for r in results:
        categories[r.label] = categories.get(r.label, 0) + 1

    logger.info(
        "Wrote %d ground truth records to %s (%.1fs)", len(results), out_path, elapsed
    )
    for cat, count in sorted(categories.items()):
        logger.info("  %s: %d", cat, count)


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
