---
status: ratified
date: 2026-09-14
deciders: <you>
---

# 0025 Hybrid search with persisted tantivy BM25 and RRF fusion

## Context

Dense-only retrieval misses exact symbols in FastAPI questions
(`OAuth2PasswordRequestForm`, `422`, file paths). A persisted lexical
index alongside Qdrant must close that gap without breaking
document-level eval or the v1-baseline reproduction.

## Decision

Build a persisted tantivy BM25 index at `rag-index build` time from
the same chunks indexed in Qdrant, under one corpus fingerprint.
At query time fuse dense (`k * overfetch_factor`) and sparse
(`sparse_k = 50`) chunk hits with RRF (`rrf_k = 30`, dense:sparse
`0.75 : 0.25`, normalized to sum to 1 in `rrf_fuse`), then dedupe
per ADR-0021. Hybrid is the active retriever (`hybrid_enabled: true`
in the committed config); the dense-only path stays available as a
baseline for `rag-eval diff`. Query tokens are filtered with the
canonical Snowball English stopword list (`src/app/stopwords_en.txt`)
and a 3-character minimum. Defaults:
`tantivy_index_dir = data/.rag-index/tantivy`.

## Alternatives considered

- Qdrant BM25 (spiked 2026-09-14): native BM25 needs the
  `qdrant/bm25` inference model, which requires cloud or local
  inference infrastructure absent from the self-hosted v1.15.1
  server; the qdrant-client 1.16.2 query API has no text-query
  model. Client-side sparse vectors plus server-side RRF do work,
  but require a sparse embedder and a collection migration; lost on
  infrastructure.
- rank-bm25 in-memory: zero infra but O(N) scan per query, no
  persistence, weak tokenization; lost on lifecycle.
- bm25s in-memory: faster than rank-bm25 but still process-local
  with hand-rolled persistence; lost on production shape.

## Consequences

What becomes easier? Exact-term and phrase recall becomes measurable
via `rag-eval diff v1-baseline v2-hybrid`. Harder? Two indexes must
stay in sync; adds the compiled `tantivy` dependency.

## Rationale

Tantivy won because it is a persisted engine, not a scoring
function.

Persisted inverted index (mmap reads, no per-process rebuild);
stemming plus fielded and phrase queries; embedded library with no
extra service; proven Quickwit core with Python bindings.

RRF needs no score calibration between dense cosine and BM25.
Distribution (S3, manifest, versioning) is deferred to 0026.
