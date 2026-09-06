# Eval Harness — Plan

> Source: `notes/architecture.md`. Scope: A (ground truth) + B (harness) + D (baseline adapter). RAG iterations out of scope.
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
- [x] A3b incremental JSONL writes + resume-on-crash + `--dry-run` + legacy JSON-array truncation (current commit; 4/70 records produced during testing — partial artifact committed only to exercise the resume path, not as ground truth)
- [ ] A4 acceptance: `data/ground_truth.jsonl` exists and validates; counts ≈ 38/17/15; commit (deferred — full LLM-judge run out of scope for this commit)

## Phase B — Harness (`src/eval_harness/`)

- [x] B1 `adapter.py` — `RetrievalResult` + `RetrieverAdapter` Protocol
- [x] B2 `loader.py` — `GroundTruthLoader` + validation
- [x] B3 `metrics.py` — `recall_at_k`, `precision_at_k`, `aggregate_by_category`
- [x] B4 `store.py` — `ResultStore` SQLite CRUD (schema §4.4; tests use `tmp_path`)
- [x] B5 `config.py` — `HarnessConfig` (OmegaConf), `.rag-eval.yaml` loader
- [x] B6 `runner.py` — `EvalRunner` (ThreadPool, timing, recoverable/fatal handling)
- [ ] B7 `cli.py` — `rag-eval run|diff|list`, exit codes 0/1/2, `[project.scripts]`
- [x] B8 `.rag-eval.yaml` committed (tracked)
- [ ] B9 acceptance: dummy run → SQLite; `diff` two runs; commit (blocked on B7)

## Phase D — Baseline adapter (`src/app/`)

- [ ] D1 `chunker.py` — naive ~500-token, no overlap
- [ ] D2 `indexer.py` — embed chunks, store vectors + chunk→doc map
- [ ] D3 `adapter.py` — `LocalRetriever` (top-k cosine, dedupe to docs)
- [ ] D4 adapter config (chunk size, model, index path)
- [ ] D5 first run `--tag v1-baseline`
- [ ] D6 acceptance: 70 queries; `diff` renders; commit
