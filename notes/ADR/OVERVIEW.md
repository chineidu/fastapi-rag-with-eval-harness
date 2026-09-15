---
generated: 2026-09-15
covers: 0001-0027
---

# Architecture overview

A retrieval evaluation harness for RAG systems, developed against a
pinned FastAPI documentation corpus. A portable harness measures
document-level retrieval quality, and one project-specific adapter
connects it to the retriever in this repo, now running hybrid
lexical-plus-dense search (tantivy BM25 fused with dense via RRF)
over the original dense baseline, with a proposed generation contract
for grounded answers.

## Evaluation methodology

Binary relevance and recall@k are the base of the eval. Aggregation is
per difficulty category and overall, and unanswerable queries are kept
out of the metric instead of scored as zeros.

- `0001` Binary relevance with recall@k (ratified): relevant or not, no
  graded labels; recall@k is primary, and precision@k, MRR, and NDCG
  remain derivable from the same per-query data.
- `0009` Unanswerable queries (ratified): written with
  `answerable: false` and empty `relevant_docs`; kept out of recall
  means and reported as a separate corpus-coverage count.
- `0017` Per-category metric aggregation (ratified): `DIRECT_LOOKUP`,
  `MULTI_HOP`, and `CONCEPTUAL` means plus an unweighted overall mean;
  the diff command reports both levels.

## Harness architecture

The harness is portable: a small module set, a bounded-concurrency
runner, classified retries, and SQLite persistence.

- `0003` Harness and adapter separation (ratified): the harness
  (`src/eval_harness/`) knows only the adapter contract; the adapter is
  the single per-project piece.
- `0005` ThreadPoolExecutor with bounded concurrency (ratified):
  default pool of 3, configurable; the harness times each adapter call
  with `time.monotonic()`.
- `0006` SQLite run history (ratified, amended by 0026): runs and
  per-query results in one database; tags describe the experiment and
  are made unique per run by 0026.
- `0013` Harness internal module split (ratified): EvalRunner,
  MetricsCalculator, ResultStore, and GroundTruthLoader, plus adapter,
  config, and cli modules.
- `0014` Error classification with backoff (ratified): recoverable
  errors retry twice with exponential backoff; fatal errors are logged,
  skipped, and mark the run partial.
- `0026` Unique run tags with timestamp suffixes (proposed, amends
  0006): a colliding tag gets a UTC `-HHMMSS` suffix, with `-2`, `-3`,
  ... on same-second collisions; the first run under a tag keeps the
  plain name.

## Adapter contract

The harness-to-RAG boundary is document-level: the adapter owns
chunking and returns typed document hits; the harness owns timing.

- `0004` RetrieverAdapter protocol (ratified, amended by 0019 and
  0027): separate `retrieve()` and `generate()` methods; the element
  shape is amended by 0019 and the generation side is amended by 0027.
- `0012` Chunking is adapter-internal (ratified): the harness never
  sees chunks, so chunking experiments cannot invalidate ground truth.
- `0019` RetrievedDocument element type (ratified, amends 0004):
  results are `list[RetrievedDocument(doc_path, score)]` in Python and
  JSON, replacing 0004's tuples for the element shape.

## Corpus and ground truth

The eval data is a pinned corpus, real user questions plus constructed
multi-hop backfill, and LLM-judged labels over a pooled candidate set,
all identified by file paths.

- `0002` FastAPI v0.140.0 corpus (ratified): 615 files pinned at commit
  `255b912`; 154 English markdown pages and 461 Python examples, with
  translations excluded.
- `0007` 70 evaluation queries (ratified, amended by 0024): 40 GitHub
  discussion threads and 30 StackOverflow threads, split 38/17/15
  across the difficulty categories.
- `0008` JSONL ground truth format (ratified): one JSON object per line
  for crash-safe incremental writes, resume, and per-record validation.
- `0010` Sparse binary relevance via top-30 pooling (ratified): the LLM
  judge labels only the top-30 semantic-search candidates per query,
  2,100 judgments instead of 43,050.
- `0011` File paths as document identifiers (ratified): readable and
  collision-free, with no registry needed; slugs and hashes lost.
- `0022` Document IDs are ROOT-relative (ratified): one ID space shared
  by index, adapter, and ground truth; fixed 45 markdown refs that
  previously could never match.
- `0024` Constructed multi-hop backfill (ratified, amends 0007): 13
  LLM-drafted, doc-pair-bounded MULTI_HOP queries (`constructed-mh-01`
  to `13`, `source: constructed`) appended to ground truth (70 to 83
  records), labeled through the same top-30 plus judge pipeline.

## Retriever pipeline

The RAG side under test: naive chunking, a pluggable vector store,
document-level over-fetch on retrieval, and fingerprint-gated index
rebuilds, plus hybrid lexical search fused with dense.

- `0018` Naive fixed-size chunker (ratified): 2000-character chunks
  with sliding-window overlap; deterministic and dependency-free, but
  structure-aware splitting is deferred.
- `0020` Qdrant as vector store (ratified): self-hosted Qdrant behind a
  `VectorStore` protocol and factory, selected by config so backends
  stay swappable.
- `0021` Baseline retriever with over-fetch (ratified): fetch
  `k * overfetch_factor` chunks (default 5) and keep the best hit per
  document, so recall@k measures document ranking, not chunk
  clustering.
- `0023` rag-index CLI and fingerprint idempotency (ratified): `build`
  and `inspect` subcommands; a corpus fingerprint in the meta point
  lets an unchanged corpus skip embed and upsert, while a mismatch or
  `--force` rebuilds and clears stale points.

## Hybrid retrieval

A persisted lexical index runs alongside dense search, and the two are
fused before dedupe to recover exact-token matches the baseline misses.

- `0025` Hybrid search with tantivy BM25 and RRF (ratified): a
  persisted tantivy index built from the same chunks under one corpus
  fingerprint; query-time RRF fusion of dense and sparse chunk hits
  (`sparse_k = 50`, `rrf_k = 30`, dense:sparse ratio `0.75 : 0.25`
  normalized in `rrf_fuse`); hybrid is the active retriever
  (`hybrid_enabled: true`) with the dense-only path kept as the diff
  baseline; distribution across replicas deferred to a future ADR.

## Generation

Few-shot grounded answers over best-chunk texts, typed for future
answer eval, which stays deferred for now.

- `0027` Few-shot grounded generation with structured citations
  (ratified, amends 0004): OpenRouter + instructor returning
  `GeneratedAnswer` with `answer`, `citations`, `model_id`, and
  `grounded`; async-only `agenerate` over `list[SearchHit]`; prompt
  pair as `GenerationPrompt` in schemas; logic in `RAGGenerator` with
  `LocalRetriever.agenerate` delegating; generation eval deferred.

## CLI and configuration

The harness user surface is three subcommands and a committed YAML
config with CLI overrides.

- `0015` CLI surface: run, diff, list (ratified): Typer subcommands
  with exit code 1 on a threshold-exceeding regression for CI.
- `0016` OmegaConf configuration (ratified): `.rag-eval.yaml` in the
  current working directory merged with CLI flags, CLI winning;
  `--config` selects an explicit file.
