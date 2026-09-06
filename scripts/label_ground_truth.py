"""Label ground truth for the eval harness.

Loads queries from ``data/eval_dataset.jsonl``, embeds the corpus, performs
semantic search to find top-30 candidate documents per query, then uses an
LLM judge to determine which candidates are relevant.  Results are written
to ``data/ground_truth.jsonl``.

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
from anyio import Path as AsyncPath

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

    # If the judge found nothing, the query is unanswerable from this corpus.
    # Mark it explicitly instead of forcing a synthetic top-1 doc.
    answerable = bool(relevant_docs)
    if not answerable:
        logger.warning(
            "No relevant docs found for query %s; marking unanswerable",
            query.id,
        )

    return GroundTruthRecord(
        query_id=query.id,
        label=query.label if query.label else ClassificationLabel.UNKNOWN.value,
        query_text=query_text,
        relevant_docs=relevant_docs,
        answerable=answerable,
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
    out_path: Path,
    limit: int = 0,
) -> list[GroundTruthRecord]:
    """Label all queries and append results to ``out_path`` as JSONL.

    Records are appended one per line after each query completes, so a crash
    mid-run loses at most the query that was in flight. On startup, any
    ``query_id`` already present in the file is skipped — a crashed run can
    be resumed by re-invoking the script with the same ``out_path``. A legacy
    JSON-array file (the previous output format) is detected and truncated.

    Parameters
    ----------
    queries : list[UnifiedEvalRecordSchema]
        Queries to label.
    corpus_docs : list[CorpusDocument]
        Loaded corpus.
    corpus_vectors : np.ndarray
        Pre-computed corpus embeddings.
    embedder : AbstractEmbedder
        Embedder used for query vectors.
    top_k : int
        Number of candidates per query.
    concurrency : int
        Max parallel LLM judge calls per query.
    dry_run : bool
        If True, skip the LLM judge and emit placeholder ``relevant_docs``.
    out_path : Path
        Output file. Existing records are skipped on resume; a legacy
        JSON-array file is truncated before writing begins.
    limit : int
        Stop after this many queries in this run (0 = no limit). On resume,
        the count applies to queries remaining after skipping already-labeled
        query_ids, not to the total across all runs.

    Returns
    -------
    list[GroundTruthRecord]
        Records newly written in this run, in input order.

    """
    # Resume: collect query_ids already in the output file.
    done_ids: set[str] = set()
    text = ""
    aout_path = AsyncPath(out_path)
    if await aout_path.exists():
        text = (await aout_path.read_text(encoding="utf-8")).strip()
    if text:
        if text.startswith("["):
            # Legacy format from a previous version of this script.
            try:
                legacy = json.loads(text)
            except json.JSONDecodeError:
                legacy = None
            if isinstance(legacy, list):
                logger.warning(
                    "Output file %s is in legacy JSON-array format; truncating",
                    out_path,
                )
                await aout_path.write_text("", encoding="utf-8")
                text = ""
            else:
                logger.warning(
                    "Output file %s starts with '[' but is not a JSON array; ignoring",
                    out_path,
                )
        if text:
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    done_ids.add(json.loads(stripped)["query_id"])
                except json.JSONDecodeError, KeyError, TypeError:
                    logger.warning(
                        "Skipping malformed line %d in %s", line_no, out_path
                    )
    if done_ids:
        logger.info(
            "Resuming: %d queries already labeled in %s",
            len(done_ids),
            out_path,
        )

    remaining = [q for q in queries if q.id not in done_ids]
    if not remaining:
        logger.info("Nothing to label; all %d queries already done", len(queries))
        return []

    await aout_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(remaining)
    results: list[GroundTruthRecord] = []

    for i, query in enumerate(remaining, 1):
        if limit and i > limit:
            logger.info("Hit --limit %d, stopping", limit)
            break
        logger.info("[%d/%d] Labeling query %s...", i, total, query.id)
        start = time.monotonic()

        if dry_run:
            query_text = clean_query_text(query.body, query.source, query.title)
            record = GroundTruthRecord(
                query_id=query.id,
                label=query.label if query.label else ClassificationLabel.UNKNOWN.value,
                query_text=query_text,
                relevant_docs=["(dry-run)"],
                answerable=True,
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
            # Append incrementally so a crash mid-run does not lose prior work.
            async with await aout_path.open("a", encoding="utf-8") as f:
                await f.write(
                    json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                )
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
    limit: int = typer.Option(
        0, "--limit", help="Stop after N queries in this run (0 = no limit)."
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

    # 4. Label all queries (writes incrementally to out_path as JSONL)
    start: float = time.monotonic()
    asyncio.run(
        _alabel_all(
            queries,
            corpus_docs,
            corpus_vectors,
            embedder,
            top_k,
            concurrency,
            dry_run,
            out_path,
            limit,
        )
    )
    elapsed: float = time.monotonic() - start

    # 5. Summary: read back the full file so the totals reflect resume state.
    all_records: list[dict[str, object]] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                continue
            try:
                all_records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue

    categories: dict[str, int] = {}
    for record in all_records:
        label = record.get("label")
        if isinstance(label, str):
            categories[label] = categories.get(label, 0) + 1

    logger.info(
        "Ground truth file has %d records (%.1fs total)", len(all_records), elapsed
    )
    for cat, count in sorted(categories.items()):
        logger.info("  %s: %d", cat, count)


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
