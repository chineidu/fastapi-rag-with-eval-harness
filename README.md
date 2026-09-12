# fastapi-rag-with-eval-harness

<!-- TOC -->

- [fastapi-rag-with-eval-harness](#fastapi-rag-with-eval-harness)
  - [1. Quickstart](#1-quickstart)
    - [Exit codes](#exit-codes)
  - [2. Data pipeline](#2-data-pipeline)
  - [3. Eval harness](#3-eval-harness)
    - [3.1 Adapter contract](#31-adapter-contract)
    - [3.2 Metrics](#32-metrics)
  - [4. Vector index](#4-vector-index)
    - [4.1 Storage model](#41-storage-model)
    - [4.2 The meta (sentinel) point](#42-the-meta-sentinel-point)
    - [4.3 Corpus fingerprint](#43-corpus-fingerprint)
  - [5. Project layout](#5-project-layout)
  - [6. Development](#6-development)

<!-- /TOC -->

Retrieval eval harness for RAG over the FastAPI docs. It quantifies the
before/after impact of pipeline changes (chunking, embeddings, top-k) with
reproducible recall@k scores, per-category breakdowns, and regression diffs.

Two pipelines share one ground truth file:

- **Labeling (one-time)** - semantic search narrows 615 corpus docs to top-30
  candidates per query, an LLM judge picks the relevant docs, output is
  `data/ground_truth.jsonl` (70 queries: 39 answerable, 31 unanswerable).
- **Eval (every run)** - `rag-eval run` sends each query through a
  `RetrieverAdapter`, scores recall@k against ground truth, stores results in
  SQLite; `rag-eval diff` compares two runs.

## 1. Quickstart

Requires Python 3.14+ and `uv`.

```bash
uv sync
cp .env.example .env  # if present, else create .env with keys below
uv run rag-eval --help
```

```bash
# Evaluate with defaults from .rag-eval.yaml
uv run rag-eval run --tag baseline

# Override config on the CLI (CLI wins over file)
uv run rag-eval run --adapter src.eval_harness.adapter:MyRetriever --k 10 --tag chunk-512

# Compare two runs, list history
uv run rag-eval diff baseline chunk-512
uv run rag-eval list --limit 20
```

### Exit codes

For `run`: 0 complete, 1 partial (some queries failed), 2 usage/config error.

For `diff`: 0 no significant regression, 1 regressed beyond threshold, 2 usage/run-not-found error.

## 2. Data pipeline

Raw questions come from real FastAPI users (GitHub Discussions +
Stack Overflow), then get normalized, difficulty-labeled, and judged
against the corpus. See `scripts/README.md` for full option tables.

```bash
# 1. Fetch raw QA threads (needs GITHUB_READ_ACCESS, optional STACK_EXCHANGE_READ_ACCESS)
make fetch-data

# 2. Merge into one unified dataset
uv run -m scripts.normalize_eval_data normalize

# 3. Label difficulty with an LLM via OpenRouter (needs OPENROUTER_API_KEY)
uv run -m scripts.classify_eval_data classify

# 4. Judge relevance per query against docs/fastapi corpus, write ground truth
uv run -m scripts.label_ground_truth --help
```

Outputs:

| File | Contents |
|---|---|
| `data/fastapi_discussions.jsonl` | Raw GitHub threads |
| `data/fastapi_stackoverflow.jsonl` | Raw Stack Overflow threads |
| `data/eval_dataset.jsonl` | Unified, unlabeled queries |
| `data/eval_dataset_labeled.jsonl` | Queries + `DIRECT_LOOKUP` / `MULTI_HOP` / `CONCEPTUAL` label |
| `data/ground_truth.jsonl` | Queries + `relevant_docs` paths + `answerable` flag (harness input) |

Corpus: `docs/fastapi/docs/en/docs` (154 markdown pages) +
`docs/fastapi/docs_src` (461 Python snippets).

Ground truth record shape:

```json
{
  "query_id": "D_kwDOCZduT84AbRBB",
  "label": "DIRECT_LOOKUP",
  "query_text": "how do I upload a file with form data?",
  "relevant_docs": ["docs/en/docs/tutorial/request-forms.md"],
  "answerable": true
}
```

Unanswerable queries (`answerable: false`, `relevant_docs: []`) are real
questions the corpus cannot answer. They are excluded from recall means and
reported separately, so they never show as phantom regressions.

## 3. Eval harness

### 3.1 Adapter contract

The harness only depends on one interface. Implement it once per project in
`src/eval_harness/adapter.py` terms:

```python
class RetrieverAdapter(Protocol):
    def retrieve(self, query: str, k: int = 10) -> RetrievalResult: ...
```

`RetrievalResult.documents` is a ranked list of `RetrievedDocument` (`doc_path`, `score` pairs).
Chunk-to-doc mapping is the adapter's job; the harness scores at doc level.
`generate()` is reserved for future generation eval.

Point the harness at your class with an import path:

```yaml
# .rag-eval.yaml (committed; CLI flags override every key)
adapter: "app.adapter:Retriever"
ground_truth: data/ground_truth.jsonl
db: data/.rag-eval/runs.db
defaults:
  k: 10
  concurrency: 3
diff:
  threshold_absolute: 0.05
  threshold_relative: 5
```

### 3.2 Metrics

- `recall@k = |retrieved[:k] intersect relevant| / |relevant|` - "did we find
  the right docs?"
- `precision@k = |retrieved[:k] intersect relevant| / k` - "how much of the
  top-k was relevant?"
- Aggregation: mean per category (`DIRECT_LOOKUP`, `MULTI_HOP`,
  `CONCEPTUAL`) plus an `OVERALL` mean over answerable queries only.
- `diff` flags a category when `|delta|` exceeds the absolute threshold or
  `|delta%|` exceeds the relative threshold.

Full design rationale lives in `notes/ADR/`.

## 4. Vector index

The app side of the pipeline. `rag-index build` chunks the corpus, embeds
each chunk, and stores the vectors in Qdrant; `rag-index inspect` reports
what the collection holds. The eval harness reads this index through
`LocalRetriever` (`src/app/adapter.py`).

```bash
uv run rag-index build          # defaults to both corpus roots; skips when unchanged
uv run rag-index build --force  # drop and rebuild the collection
uv run rag-index inspect
```

```text
Collection fastapi_docs (qdrant)
  status:      indexed
  model_id:    BAAI/bge-small-en-v1.5
  dim:         384
  chunk_count: 1200
```

### 4.1 Storage model

Qdrant stores **points**, the analogue of database rows: an `id`, a
`vector`, and a `payload` (arbitrary JSON metadata). The index writes one
point per chunk; its id is a deterministic UUID, its vector is the
embedding, and its payload carries `chunk_id`, `doc_path`, `chunk_index`,
and `text`.

- **Collection** - the table of points, here `fastapi_docs`.
- **Upsert** - insert-or-overwrite by id, the write primitive used by the
  indexer.
- **uuid5** - a deterministic UUID derived from a name string. Chunks are
  stored under `uuid5(chunk_id)`, so re-indexing overwrites the same points
  instead of duplicating them.

### 4.2 The meta (sentinel) point

One point in the collection is not a chunk. It carries the index
bookkeeping:

```json
{
  "is_meta": true,
  "model_id": "BAAI/bge-small-en-v1.5",
  "dim": 384,
  "corpus_fingerprint": "9f3c..."
}
```

A **sentinel** is a reserved marker meaning "not regular data." It exists
because Qdrant has no collection-level metadata slot; storing it as a point
means the index state travels with the data. Its vector is a throwaway
`[0.0] * dim` (every point needs one) and its id is
`uuid5("__index_meta__")`, so the next run can read it back. Search and the
chunk count filter `is_meta == true`, so the sentinel never appears as a
retrieval hit or inflates the count.

`ensure_collection` reads the sentinel to decide reuse versus rebuild, and
`rag-index inspect` reads it for the model and dimension.

### 4.3 Corpus fingerprint

The fingerprint answers one question: is this index the index of this
exact corpus? It is a SHA-256 digest over the sorted chunk ids and the
hash of each chunk's text. Because chunk ids encode `doc_path#index`, the
digest covers the file set, the file contents, and the chunking parameters
in one short string; sorting makes it independent of chunk order.

- Match (and no `--force`) - the collection is reused and `build` skips
  embedding and upsert, printing `Index up to date`.
- Mismatch or `--force` - the collection is dropped and rebuilt, which
  also clears stale points left by deleted or shrunk documents.

Think of it as a cache key, `model_id + dim + corpus_fingerprint`, not a
diff: it cannot tell you *what* changed. Design rationale in ADR-0023 (and
ADR-0020 for the store itself).

## 5. Project layout

```text
.
├── src/
│   ├── eval_harness/      # Copyable harness: cli, runner, metrics, store, loader, adapter, config
│   ├── app/               # RAG side: chunker, indexer, Qdrant store, adapter, rag-index CLI
│   ├── embeddings/        # AbstractEmbedder + local (fastembed) / api (OpenRouter) / stub
│   ├── schemas/           # Pydantic models, harness dataclasses, StrEnum types
│   ├── config/            # pydantic-settings (ENV/HOST/PORT + pipeline YAML in config.yaml)
│   ├── prompts/           # Reusable LLM prompts for labeling/classification
│   └── utils/             # JSONL io, HTML strip, text cleaning
├── scripts/               # One-off pipeline CLIs: fetch, normalize, classify, label
├── tests/                 # Mirrors src/, pytest with 85% coverage gate
├── data/                  # Eval JSONL files + gitignored .rag-eval/runs.db
├── docs/fastapi/          # FastAPI repo snapshot used as retrieval corpus
├── notes/ADR/             # Architecture decision records
├── .rag-eval.yaml         # Harness defaults (adapter, k, db, diff thresholds)
└── Makefile               # install / test / lint / format / typecheck / check
```

## 6. Development

```bash
make install       # uv sync
make test          # pytest, concise
make test-cov      # pytest with coverage report
make lint          # ruff check
make format        # ruff check --fix + ruff format
make typecheck     # ty check
make check         # lint + typecheck + test (must pass before commit)
```

Conventions from `AGENTS.md`: `uv run` for everything, Ruff line length
110, NumPy docstrings, `StrEnum` over raw strings, no `print()` in library
code (use module loggers), `slow` pytest marker for tests that download
models or hit the network (deselected by default).
