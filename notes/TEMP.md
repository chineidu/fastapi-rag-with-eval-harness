# Eval Harness — Plan

> Source: `notes/ADR/`. Scope: A (ground truth) + B (harness) + D (baseline adapter). RAG iterations out of scope.
>
> Per phase: `make check`, one commit. Out of scope: generation eval, latency/cost, CI, chunking v2/hybrid/rerank/prompt tuning.

## Decisions

- LLM judge: OpenRouter + instructor (reuse `classify_eval_data.py` pattern)
- Embeddings: local default `BAAI/bge-small-en-v1.5` via fastembed (model_id configurable) + optional `ApiEmbedder` via openai SDK → OpenRouter `/embeddings` (reuse `OPENROUTER_*` creds; default model `openai/text-embedding-3-small`)
- CLI: `rag-eval` (run/diff/list) via `[project.scripts]`; per-query timeout 30s default

## Phase A — Ground truth

- [x] A0 verify corpus — 154 EN `.md` + 461 `docs_src/*.py` (FastAPI v0.140.0, `255b912`)
- [x] A1 `clean_query_text()` in `src/utils/text.py` (commit `2b185ae`)
- [x] A2 `src/embeddings/` — `AbstractEmbedder` (ABC) + `LocalEmbedder` + `ApiEmbedder` + `StubEmbedder` + `make_embedder` factory + config wiring + tests (renamed from `EmbedderPort` Protocol in `fba7111`)
- [x] A3 `scripts/label_ground_truth.py` — embed corpus (cache `.np`) → top-30 cosine → LLM judge → confidence gates → JSONL output; add `src/schemas/ground_truth.py` (pipeline + schema shipped in `82b57b8`)
- [x] A3b incremental JSONL writes + resume-on-crash + `--dry-run` + legacy JSON-array truncation + `--limit` (commit `9a8e089`)
- [x] A4 acceptance: `data/ground_truth.jsonl` exists and validates; counts ≈ 38/17/15; commit (70 records: 39 answerable + 31 unanswerable; labels 38/17/15;
**unanswerable queries marked `answerable: false` with empty `relevant_docs` instead of synthetic top-1 fallback** — the 31 fallbacks were mostly GitHub bug reports/regressions not answerable from docs.
`GroundTruthRecord` gained `answerable` field with consistency validator; runner excludes unanswerable from recall@k means and reports `unanswerable` count in `RunSummary`; `top_k` default stays 30)

## Phase B — Harness (`src/eval_harness/`)

- [x] B1 `adapter.py` — `RetrievalResult` + `RetrieverAdapter` Protocol
- [x] B2 `loader.py` — `GroundTruthLoader` + validation
- [x] B3 `metrics.py` — `recall_at_k`, `precision_at_k`, `aggregate_by_category`
- [x] B4 `store.py` — `ResultStore` SQLite CRUD (schema §4.4; tests use `tmp_path`)
- [x] B5 `config.py` — `HarnessConfig` (OmegaConf), `.rag-eval.yaml` loader
- [x] B6 `runner.py` — `EvalRunner` (ThreadPool, timing, recoverable/fatal handling)
- [x] B7 `cli.py` — `rag-eval run|diff|list`, exit codes 0/1/2, `[project.scripts]` (**`diff` must exclude unanswerable queries from means, like `runner.py` does — they sit at recall 0.0 in the store and would show phantom regressions**)
  - typer CLI in `src/eval_harness/cli.py`; `diff` excludes rows with empty `ground_truth` JSON (the unanswerable marker), matching runner semantics
  - `[project.scripts]` wired via hatchling `[build-system]` (`packages = ["src"]`); `uv run rag-eval` works (`python -m src.eval_harness.cli` also works)
  - added `--config` flag (not in architecture.md) to point at an explicit `.rag-eval.yaml`; `run` exits 1 on partial runs
  - typer added as explicit dependency via `uv add typer` (was transitive via fastapi[standard])
- [x] B8 `.rag-eval.yaml` committed (tracked)
- [x] B9 acceptance: dummy run → SQLite; `diff` two runs; commit
  - throwaway `EmptyAdapter` (recall 0.0) vs `OracleAdapter` (recall ~0.98) over all 70 queries; both `complete` 70/70 in `data/.rag-eval/runs.db`
  - `diff` both directions flags improved/regressed per category; exit 1 on regression, 0 on improvement; `list` renders both runs

## Phase D — Baseline adapter (`src/app/`)

- [x] D1 `chunker.py` — naive ~500-token, no overlap
- [x] D2 `indexer.py` — embed chunks, store vectors + chunk→doc map
- [x] D3 `adapter.py` — `LocalRetriever` (top-k cosine, dedupe to docs)
- [x] D4 adapter config (chunk size, model, index path)
  - realized as `retriever_config.overfetch_factor` (default 5, ADR-0021); model/chunk settings reused from `embeddings_config` + `indexer_config` so query vectors cannot drift from the indexed ones
  - `.rag-eval.yaml` adapter path fixed to `src.app.adapter:LocalRetriever` (previous `app.adapter:Retriever` did not import)
- [x] D5 first run `--tag v1-baseline`
  - prerequisites fixed before the run: GT md refs normalized to ROOT-relative (ADR-0022), `label_ground_truth._load_corpus` aligned.
  `Indexer.build`/`rag-index --corpus` extended to index both corpus roots in one pass (1200 chunks; 1201 points incl. meta sentinel; `fastapi_docs`, 384-dim cosine)
  - run 3: 70/70 complete, 31 unanswerable; recall@10 OVERALL 0.627 (DIRECT_LOOKUP 0.706, MULTI_HOP 0.479, CONCEPTUAL 0.506)
- [x] D6 acceptance: 70 queries; `diff` renders; commit
  - run 4 `v1-baseline-repeat` identical per category; `diff v1-baseline v1-baseline-repeat` renders zero deltas, exit 0
  - ADR-0020, ADR-0021, and ADR-0022 ratified 2026-09-11 and committed in `c91165b`
  - qdrant-client pinned to `>=1.16,<1.17` (1.16.2) against server v1.16.3 (`docker-compose.yml`), so both sit on the 1.16 minor.
  `==1.15.1` exact-matched the old server pin but leaked sqlite ResourceWarnings in tests; run 5 `v1-baseline-check` matches the baseline exactly
  - 2026-09-12: `diff` across runs 3/4/5 renders +0.000 for every category and OVERALL (31/31 unanswerable rows excluded), exit 0; D6 accepted because the `diff` command already met the criterion
