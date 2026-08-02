# Eval Harness — Architecture Reference

> Brainstorm session date: 2026-08-02

> Status: design phase (brainstorm complete, ready for plan mode)

---

## 1. Purpose

Quantifiably measure the impact of changes to a RAG system — before vs after — with reproducible, comparable results. The primary user is a senior engineer evaluating trade-offs when modifying a RAG pipeline.

---

## 2. Scope

### In scope (v1)
- **Retrieval quality** — binary recall@k, per-category breakdown + delta from baseline

Binary relevance: each document is either relevant or not (0 or 1), no intermediate scores. This is simpler and faster to label than graded relevance (e.g., "essential" vs "somewhat useful").

**Why recall@k as the starting metric?** It answers the most fundamental retrieval question: "did we find the right docs?" It is interpretable, standard in information retrieval, and doesn't require a generator/LLM. Precision@k, MRR, and NDCG are derivable from the same data once recall@k is working. Binary relevance is faster to label than graded relevance. Starting with the simplest meaningful metric avoids premature complexity.

### Explicitly deferred (extensible, not designed)
- Generation quality evaluation (faithfulness, correctness, relevance)
- Latency / cost tracking
- The adapter already defines `generate()` as a future method; the SQLite schema reserves `metrics_extra` JSON for new metrics.

---

## 3. Data Model

### 3.1 Corpus

615 FastAPI documentation files from the FastAPI repo source tree (v0.140.0 snapshot, upstream commit `255b912`, 2026-07-24; verified 2026-08-02):

| Source | Count | Description |
|---|---|---|
| `docs/fastapi/docs_src/*.py` | 461 | Python example code snippets |
| `docs/fastapi/docs/en/docs/*.md` | 154 | Markdown documentation pages (EN only; 12 translation trees excluded by design) |

### 3.2 Evaluation Queries

70 real user questions sourced from GitHub discussions and StackOverflow:

| Source | Count | Format |
|---|---|---|
| `data/fastapi_discussions.jsonl` | 40 | GitHub discussion threads |
| `data/fastapi_stackoverflow.jsonl` | 30 | StackOverflow Q&A threads |

Difficulty categories:
- `DIRECT_LOOKUP` (38) — answer exists in a single doc page
- `MULTI_HOP` (17) — answer requires synthesising across multiple doc pages
- `CONCEPTUAL` (15) — requires understanding concepts spread across multiple pages

### 3.3 Ground Truth Schema

Format: JSON array, one object per query.

```json
{
  "query_id": "D_kwDOCZduT84AbRBB",
  "label": "DIRECT_LOOKUP | MULTI_HOP | CONCEPTUAL",
  "query_text": "<the question text the harness sends to the retriever>",
  "relevant_docs": [
    "docs/en/docs/tutorial/request-forms.md",
    "docs/en/docs/tutorial/body.md"
  ],
  "source": "github | stackoverflow",
  "title": "<human-readable title of the original question>",
  "answer_text": "<reference answer, not used by retrieval eval>",
  "url": "<link to original GitHub/SO post>"
}
```

| Field | Required | Purpose |
|---|---|---|
| `query_id` | ✅ | Unique, matches original GitHub/SO discussion/answer ID |
| `label` | ✅ | Difficulty category; drives per-category breakdown in `diff` |
| `query_text` | ✅ | What the harness passes to `adapter.retrieve()` |
| `relevant_docs` | ✅ | List of file paths to FastAPI documentation pages that answer this query |
| `source` | ❌ | Origin of the question (`github` or `stackoverflow`) |
| `title` | ❌ | Human-readable, aids manual review during labeling |
| `answer_text` | ❌ | Reference answer; context for labeling, not used in retrieval eval |
| `url` | ❌ | Link to original source |

**Labeling assumption:** Sparse binary relevance. Any doc not listed in `relevant_docs` is treated as irrelevant. This is the standard IR pooling assumption.

**Why sparse?** Complete labeling (70 queries × 615 docs = 43,050 judgments) is infeasible for a solo engineer.
Semantic search narrows each query to ~30 candidates, then the LLM judges those 30 (70 × 30 = 2,100 judgments total).
Of those, most queries have 1–3 relevant docs (DIRECT_LOOKUP typically 1, MULTI_HOP and CONCEPTUAL 2–5), producing ~140–210 total relevant labels.
The pooling assumption (standard in TREC, BEIR) says: docs that never surfaced in the top-30 semantic search are assumed irrelevant.
This biases recall@k slightly upward (you miss some relevant docs), but the *relative* comparison (before vs after) remains valid since both runs are evaluated against the same ground truth.

**Doc identifiers:** File paths (e.g. `docs/en/docs/tutorial/body.md`). The adapter internally derives slugs from paths; the harness only operates on paths.

**Why file paths?** Paths are stable, universal, and unambiguous in a file-based corpus. Slugs can collide across directories or change when content is reorganized. Paths are the canonical identifier; slugs are a convenience layer on top.

### 3.4 Labeling Pipeline

All 70 queries need labeling against the 615-doc corpus. Workflow:

```
For each query:
  1. Semantic search (query_text → docs) → top-30 candidate docs
  2. For each candidate:
     LLM judges relevance given query_text + answer_text + full doc content
     → {"verdict": "relevant|irrelevant", "rationale": "...", "confidence": 0.0-1.0}
  3. Human review of extreme confidence values:
     - confidence ≥ 0.9  → spot-check 10%
     - confidence ≤ 0.3  → full human review
     - 0.3 < confidence < 0.9 → accepted as-is
```

The labeling script lives in `scripts/` (e.g. `scripts/label_ground_truth.py`). It is a one-time utility — not part of the harness.

---

## 4. Architecture

### 4.1 High-Level Separation

```
Harness (copyable module)  ──► Adapter (project-specific)
                                  implements RetrieverAdapter protocol
```

- **Harness** (`src/eval_harness/`): runner, metrics engine, CLI, SQLite persistence, diff command. Knows how to evaluate *any* adapter that follows the contract.
- **Adapter** (project-specific): bridges the harness to a specific RAG codebase. Written once per project, implements `RetrieverAdapter`.
- The harness module is copyable to future repos; the adapter is the only per-project code.

**Why an adapter boundary?** This is the single decision that makes the harness reusable across projects. Without it, the harness would be coupled to one RAG codebase and unportable. The adapter pattern is a standard design pattern for inversion of control. It also enables testing the harness itself with a dummy adapter that returns known results.

### 4.2 Adapter Contract

```python
@dataclass
class RetrievalResult:
    documents: list[tuple[str, float]]  # [(doc_path, score), ...]
    metadata: dict[str, object]  # chunking info, model name, config hash, etc.


class RetrieverAdapter(Protocol):
    def retrieve(self, query: str, k: int = 10) -> RetrievalResult: ...
    def generate(self, query: str, documents: list[str]) -> str: ...  # deferred
```

Key design decisions:

- `retrieve()` and `generate()` are **separate methods** — the harness calls them independently. Retrieval and generation are independent concerns. You can evaluate retrieval without triggering an expensive LLM call. The harness can call `retrieve()` for retrieval eval today and add `generate()` for generation eval later — the retrieval path is untouched. This also means the adapter contract works for retrieval-only RAG systems that have no generator.

- The harness **owns timing measurement** — it wraps adapter calls with `time.monotonic()`. If each adapter self-reports timing, measurements may differ in methodology or be omitted entirely. The harness wrapping ensures consistent, trustworthy timing across all adapters. The adapter can still report internal breakdowns (embedding time, LLM time) in `metadata`.

- The adapter abstracts away **chunking** — the harness operates at the document level. Chunk-to-doc deduplication is the adapter's responsibility. Chunk metadata goes in `metadata` for auditability. Chunking is an implementation detail of the RAG system, not an eval concern. The harness asks: "which documents were relevant?" and "which documents were retrieved?" How the adapter maps retrieved chunks back to documents is its own concern. This keeps the harness simple and the ground truth stable — doc identifiers don't change when chunking strategy changes.

- Doc identifiers are **file paths**. The adapter internally derives slugs if needed.

### 4.3 Harness Internals

```
EvalRunner          — orchestrates queries, concurrency, error handling
MetricsCalculator   — pure functions: recall_at_k, precision_at_k, aggregate_by_category
ResultStore         — SQLite CRUD: create_run, save_result, finalize_run, query for diff
GroundTruthLoader   — loads + validates ground truth JSON
```

#### Runner flow

```
EvalRunner.run(tag)
  → ResultStore.create_run(tag)
  → for each query in ground_truth (bounded concurrency, default 3):
      → adapter.retrieve(query_text, k)
      → MetricsCalculator.recall_at_k(retrieved, relevant_docs, k)
      → ResultStore.save_result(run_id, query_result)
  → MetricsCalculator.aggregate_by_category(all_results)
  → ResultStore.finalize_run(run_id, "complete" | "partial")
  → return per-category scores
```

#### Concurrency

- `concurrent.futures.ThreadPoolExecutor` with bounded pool size (default: 3)
- Each adapter call is individually timed
- Configurable via `.rag-eval.yaml` key `defaults.concurrency` or CLI `--concurrency`

**Why ThreadPoolExecutor over asyncio?** Adapters may be sync or async. `ThreadPoolExecutor` works with both; wrapping async with `asyncio.to_thread()` is straightforward. An async event loop adds complexity for a tool that runs 70 queries — not a high-throughput server. Threads are simpler to reason about, debug, and test.

**Why bounded concurrency (default 3)?** Unlimited concurrency can overwhelm the adapter (e.g., loading an embedding model into memory per call, or hitting API rate limits). Sequential execution is too slow for 70 queries. 3 is a reasonable default that balances speed and safety; configurable per adapter. Individual call timing remains clean since each call is independently measured.

#### Error Handling

| Error class | Examples | Behaviour |
|---|---|---|
| **Recoverable** | `ConnectionError`, `TimeoutError`, HTTP 5xx | Retry up to 2 times with exponential backoff |
| **Fatal** | `ValueError`, `TypeError`, malformed results | Log + skip, mark query as `"error"` |

- Failed queries are counted and displayed: `"✓ 67/70 completed, ✗ 3 failures"`
- Incomplete runs get status `"partial"` (not `"complete"`)
- Diff warns if either run is incomplete

**Why recoverable vs fatal?** A transient network error (`ConnectionError`) will likely succeed on retry; a type error (`TypeError`) in the adapter will fail every time. Without this distinction, you'd either retry everything (wasteful, and fatal errors loop until max retries) or retry nothing (fragile; a single network blip poisons the run).

**Why exponential backoff?** Avoids thundering herd on rate-limited services and is standard practice for transient failures.

### 4.4 SQLite Schema

```sql
-- Database path: data/.rag-eval/runs.db (gitignored, configurable)

CREATE TABLE runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tag         TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime()),
    status      TEXT NOT NULL DEFAULT 'running',   -- 'running', 'complete', 'partial'
    config      TEXT,                               -- JSON: {"k": 10, "concurrency": 3, "adapter": "..."}
    metadata    TEXT                                -- JSON: freeform notes
);

CREATE TABLE query_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    query_id           TEXT NOT NULL,                -- matches ground truth query_id
    category           TEXT NOT NULL,                -- DIRECT_LOOKUP, MULTI_HOP, CONCEPTUAL
    k_value            INTEGER NOT NULL,
    retrieved_docs     TEXT NOT NULL,                -- JSON: [["path", 0.95], ...]
    ground_truth       TEXT NOT NULL,                -- JSON: ["path_a", "path_b"]
    recall_at_k        REAL,
    precision_at_k     REAL,
    latency_ms         REAL,
    status             TEXT NOT NULL DEFAULT 'success',  -- 'success', 'recoverable', 'fatal'
    error              TEXT,                              -- error message if failed
    metrics_extra      TEXT                               -- JSON: chunk counts, future metrics
);

CREATE INDEX idx_query_results_run_query ON query_results(run_id, query_id);
```

Design notes:
- `retrieved_docs` stores the full ranked list (not just the metric) for debugging — enables per-query drill-down on regressions.
- `metrics_extra` is a JSON catch-all for experimental metrics. Promote to columns when stable.
- Tags are **non-unique** — multiple runs can share a tag; `diff` resolves to the latest run with that tag. Use `--run-id` for precise comparison.

**Why SQLite over JSON files?** SQLite enables querying across runs: "show me all MULTI_HOP scores from the last 30 days." A JSON-per-file approach would require a directory scan and manual parsing. SQLite is serverless, a single file, requires zero setup — ideal for a local dev tool. The file is gitignored; only the harness code and ground truth live in version control. At ~70 queries per run, the database stays small.

**Why non-unique tags?** Tags describe *what kind of run* this is ("chunking experiment"), not *which specific run this is*. You might run the same config 3 times to measure variance, all tagged `chunk-size-512`. This follows the pattern used by MLflow and W&B.

### 4.5 CLI Surface

```
rag-eval run [OPTIONS]
rag-eval diff TAG|ID TAG|ID [OPTIONS]
rag-eval list
```

#### `rag-eval run`

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--adapter` | ✅ | — | Import path to adapter class |
| `--ground-truth` | ✅ | — | Path to enriched JSON file |
| `--tag` | ❌ | auto-generated | Label for this run |
| `--k` | ❌ | `10` | Top-k for recall@k |
| `--concurrency` | ❌ | `3` | Max parallel adapter calls |
| `--db` | ❌ | `data/.rag-eval/runs.db` | SQLite path |

#### `rag-eval diff`

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--threshold-absolute` | ❌ | `0.05` | Flag if |Δ| exceeds |
| `--threshold-relative` | ❌ | `5` | Flag if |Δ%| exceeds (percent) |
| `--verbose` | ❌ | off | Show per-query results |
| `--against latest` | ❌ | — | Compare to most recent run |
| `--run-id` | ❌ | — | Reference runs by ID instead of tag |
| `--db` | ❌ | `data/.rag-eval/runs.db` | SQLite path |

Output (default — per-category summary):

```
                  recall@10
CATEGORY          baseline   current      Δ      %
───────────────   ────────   ───────   ──────  ─────
DIRECT_LOOKUP       0.92      0.93     +0.01   +1.1%
MULTI_HOP           0.45      0.53     +0.08  +17.8%  ✓
CONCEPTUAL          0.38      0.36     -0.02   -5.3%  ✗
──────────────────────────────────────────────────
OVERALL             0.67      0.69     +0.02   +3.0%

1 regression above threshold. Run with --verbose for per-query details.
```

Output (`--verbose`): adds per-query delta rows.

Exit codes:
- `0` — no significant regressions
- `1` — at least one category regressed beyond threshold
- `2` — diff failed (missing run, bad args)

#### `rag-eval list`

```
ID   TAG                     CREATED              STATUS    QUERIES
───  ─────────────────────   ───────────────────  ────────  ───────
7    chunk-size-512          2026-01-03 12:05     complete  70/70
6    v1-baseline             2026-01-03 11:00     complete  70/70
5    chunk-size-512          2026-01-03 10:30     partial   68/70
```

### 4.6 Configuration

Format: YAML, parsed by OmegaConf. File: `.rag-eval.yaml` (project root, committed).

```yaml
# .rag-eval.yaml
adapter: myproject.adapter:LocalRetriever
ground_truth: data/ground_truth.json
db: data/.rag-eval/runs.db

defaults:
  k: 10
  concurrency: 3

diff:
  threshold_absolute: 0.05
  threshold_relative: 5
```

Merging order: file config → CLI overrides. CLI wins.

**Why OmegaConf?** Already used in this engineer's other projects — consistency matters. Structured configs (dataclass-based) give runtime type safety: `k: int = 10` can't accidentally become a string. Merge semantics (file config → CLI overrides) map directly to the harness's needs. `OmegaConf.from_cli()` automatically parses `--k 20 --concurrency 5` without manual argument parsing.

---

## 5. Metrics

### 5.1 recall@k

```
recall@k = |retrieved ∩ relevant| / |relevant|
```

"What fraction of the good stuff did you catch in the top k?"

Example: relevant docs = {A, B, C} (3 total), retrieved top-5 = [B, X, A, Y, Z].
recall@5 = 2/3 = 0.67

### 5.2 precision@k

```
precision@k = |retrieved ∩ relevant| / k
```

"Of the top k results, how many were actually relevant?"

### 5.3 Aggregation

- Per-query → per-category (mean recall@k across all queries in `DIRECT_LOOKUP`, `MULTI_HOP`, `CONCEPTUAL`)
- Overall = mean across all 70 queries

**Why per-category breakdown + deltas?** An overall metric hides regressions in specific categories. If CONCEPTUAL drops 0.10 but DIRECT_LOOKUP improves 0.05, the average looks flat — but you've broken conceptual retrieval. Per-category breakdown tells you *where* the system is weak and *what kind* of change helped. The delta answers the before/after question directly: "did this change make things better or worse?" Thresholds (absolute + relative) filter out noise from meaningful change.

---

## 6. Future Extensibility

| Extension | What's already in place |
|---|---|
| Generation quality | `adapter.generate()` method defined in the protocol; `query_results` has `ground_truth` column with answer text |
| Latency tracking | `latency_ms` column; adapter `metadata` can hold per-phase breakdowns |
| Additional metrics | `metrics_extra` JSON column in `query_results` — add new metrics without schema migration |
| New difficulty categories | Ground truth `label` field is a free string; `diff` groups on it automatically |
| CI integration | Exit codes defined; `--against latest` enables zero-config CI diff |

---

## 7. Project Layout

```
fastapi-rag-with-eval-harness/
├── src/
│   └── eval_harness/          # The harness module (copyable)
│       ├── __init__.py
│       ├── cli.py             # CLI entry point
│       ├── runner.py          # EvalRunner
│       ├── metrics.py         # MetricsCalculator
│       ├── store.py           # ResultStore (SQLite)
│       ├── loader.py          # GroundTruthLoader
│       ├── adapter.py         # RetrieverAdapter protocol + RetrievalResult
│       └── config.py          # OmegaConf structured config
├── src/
│   └── app/                   # Your RAG application (NOT part of the harness)
│       └── adapter.py         # Project-specific adapter implementation
├── data/
│   ├── eval_dataset_labeled.jsonl     # 70 labeled queries (original format)
│   ├── eval_dataset.jsonl             # 70 unlabeled queries
│   ├── fastapi_discussions.jsonl      # Raw GitHub discussions
│   ├── fastapi_stackoverflow.jsonl    # Raw StackOverflow Q&As
│   ├── ground_truth.json             # Enriched ground truth (to be created)
│   └── .rag-eval/                     # (gitignored)
│       └── runs.db                    # SQLite run history
├── docs/
│   └── fastapi/                       # FastAPI docs source (corpus)
├── scripts/
│   └── label_ground_truth.py         # LLM-assisted labeling utility
├── tests/
├── notes/
│   └── architecture.md                # This file
├── .rag-eval.yaml                     # Harness config
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 8. Decisions Not Made (Intentionally Left Open)

- Exact LLM model and provider for labeling
- Embedding model for semantic search in the labeling pipeline
- Per-query timeout duration
- CLI subcommand naming (`rag-eval` vs something else)
