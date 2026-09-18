# Iteration notes

Per-iteration findings: what changed, eval numbers before/after, and why.
Raw per-run data lives in `data/.rag-eval/runs.db`; this file holds the
interpretation. (See `notes/plan.md` Phase 3.)

## Baseline - run 1 (2026-09-13)

First run after a clean slate (previous run history wiped; DB reset so this
83-record eval set starts at run 1). Run 2 repeats run 1 exactly.

Config: naive fixed-size chunks (`chunk_size=2000`, `overlap=0`), local
`BAAI/bge-small-en-v1.5` (384-dim cosine, Qdrant collection `fastapi_docs`),
`overfetch_factor=5` with best-chunk-per-doc dedup, `k=10`.
Eval set: `data/ground_truth.jsonl`, 83 records (52 answerable, 31
unanswerable excluded from means).

Recall@10: DIRECT_LOOKUP 0.706 (n=24), MULTI_HOP 0.487 (n=17),
CONCEPTUAL 0.506 (n=11), OVERALL 0.592 (n=52).

Notes:

- MULTI_HOP went 4 -> 17 answerable via 13 constructed pair-bounded
  questions (see `notes/ADR/0024-constructed-multi-hop-backfill.md`). New
  rows average 0.489 vs 0.479 for the original 4: same level, now stable.
- OVERALL reads lower than the old 70-query 0.627 only by composition:
  multi-hop (weakest category) grew from 10% to 33% of scored queries.
  No per-category regression.
- `constructed-mh-03` scores 0.0 at k=10 while 6/7 judged docs rank 12-28
  at k=30: file-upload half saturates the top 10. Kept as the hard case
  that hybrid search or reranking must move.
- 31 unanswerable queries are corpus-coverage signal, kept for future
  abstention testing per ADR-0009.

## Hybrid v1 - run 4 (2026-09-14)

First hybrid iteration (ADR-0025): persisted tantivy BM25 index built
alongside Qdrant at `rag-index build --force`, fused with dense hits via
RRF (`rrf_k=60`, `sparse_k=50`, `overfetch_factor=5`, `k=10`). Run 3 was
a partial (6/83) casualty of the query-parser bug below; run 4 is the
clean measurement.

Recall@10: DIRECT_LOOKUP 0.525 (n=24), MULTI_HOP 0.530 (n=17),
CONCEPTUAL 0.545 (n=11), OVERALL 0.531 (n=52).

Notes:

- OVERALL regressed -0.061 (-10.3%) vs baseline; DIRECT_LOOKUP is the
  driver (-0.181, -25.6%). MULTI_HOP (+0.043) and CONCEPTUAL (+0.038)
  improved.
- Hypothesis: the OR-term BM25 over raw question text (long, code-heavy)
  floods fusion with token-overlapping but wrong chunks, crowding out
  dense hits that were already strong on direct-lookup queries. Exact
  terms help multi-hop/conceptual because they bridge pages dense misses.
- Bug found and fixed during the run: `parse_query` fed raw question
  text into tantivy's query language and died on code/punctuation
  (77/83 failed). Replaced with a tokenized OR query of term queries
  (`_query_tokens`), matching the default tokenizer.
- Next levers if iterating: lower `sparse_k`, raise `rrf_k`, weighted
  fusion, or stopword/min-token filtering on the lexical side.

## Hybrid tuning sweep - runs 5-11 (2026-09-14)

Swept query-time fusion params against the fixed tantivy index (no
rebuilds). Recall@10 OVERALL (baseline dense-only = 0.592):

| Config | DL | MH | CON | OVERALL |
|---|---|---|---|---|
| rrf60/sk50 (defaults) | 0.525 | 0.530 | 0.545 | 0.531 |
| rrf60/sk20 | 0.516 | 0.528 | 0.249 | 0.463 |
| rrf30/sk50 | 0.519 | 0.540 | 0.454 | 0.512 |
| +stopwords (Snowball) | 0.550 | 0.581 | 0.363 | 0.521 |
| +dense_weight 2.0 | 0.576 | 0.573 | 0.449 | 0.549 |
| +dense_weight 3.0 | 0.613 | 0.573 | 0.453 | 0.566 |
| +dense_weight 4.0 | 0.617 | 0.563 | 0.457 | 0.566 |
| w3 + rrf30 (final) | 0.665 | 0.583 | 0.457 | 0.594 |

Winner: `sparse_k=50, rrf_k=30, dense_weight=0.75, sparse_weight=0.25`
(normalized ratio, equivalent to 3:1) + Snowball stopwords + min token
len 3. OVERALL +0.002 vs baseline; MULTI_HOP +0.096 (+19.8%) is the
real win, DIRECT_LOOKUP -0.042 and CONCEPTUAL -0.049 are the cost.
Weights are normalized to sum to 1 inside `rrf_fuse`, so any ratio
representation (3:1, 0.75:0.25) scores identically.

Notes:

- Every single-lever change traded one category against another;
  only the combined w3+rrf30 broke past baseline. The sparse list is
  noisier than dense, so dense must dominate the fusion (weight 3x)
  and ranks must be sharp (rrf_k=30) for sparse to only matter where
  dense misses.
- `constructed-mh-03` is still 0.0 at k=10 (was the flagged hard
  case): hybrid did not move it, reranking remains the lever.
- Stopword list is the canonical Snowball English set, loaded from
  `src/app/stopwords_en.txt` (package data, importlib.resources).
- Tests are now hermetic: adapter tests inject an explicit config
  stub, so the committed `hybrid_enabled: true` cannot leak into
  test outcomes.

## Hybrid index workflow - option 2 (pinned manifest, versioned collections)

Provisional - from brainstorm 2026-09-13. Same image, two entrypoints.
Rare rebuilds, 2 API replicas, managed Qdrant in prod, Tantivy via S3.

1. CI builds image v123 once (has `rag-index` + API in it).
2. CI runs one-off job: `rag-index build` from image v123, writes to new
   Qdrant collection `docs_v123` (old `docs_v122` untouched).
3. Same job builds Tantivy index, pushes `s3://bucket/tantivy-v123.tar.gz`.
4. Same job writes `s3://bucket/manifest-v123.json` with collection name,
   tarball key, fingerprint, doc count.
5. Deploy sets `INDEX_VERSION=v123` on both replicas (same image v123).
6. Each replica on boot fetches manifest, downloads tarball to local disk,
   connects to `docs_v123`, verifies fingerprint. Fails readiness on mismatch.
7. Traffic shifts once both pass readiness; old replicas drain.
8. Cleanup later deletes `docs_v122` + old tarball, keeps last 1-2 for rollback.
   Rollback is redeploy with `INDEX_VERSION=v122`.

## Structural chunker v1 (2026-09-17)

ADR-0031 (proposed). `indexer_config.chunk_strategy: naive | structural`,
default naive. Structural packs whole ATX-all-levels + Setext markdown
sections (fenced code skipped) and top-level def/class Python blocks via
`ast`; oversized units fall back to naive char-slicing. No new deps.

Index: 1416 structural chunks vs 1200 naive (same corpus, same 2000/0 size).
Eval `rag-eval diff v2-hybrid-ratio-075-025 structural-v1` (recall@10):

- OVERALL 0.594 -> 0.575 (-0.020, -3.3%, noise)
- DIRECT_LOOKUP 0.665 -> 0.624 (-0.041, -6.1%, flagged regressed)
- MULTI_HOP 0.583 -> 0.566 (-0.017, -2.9%, noise)
- CONCEPTUAL 0.457 -> 0.480 (+0.022, +4.9%, noise)

Notes:

- Roughly neutral overall: boundary-respecting chunks did not move the
  needle at doc-level recall@k. Smaller average chunk size (more chunks
  for the same corpus) spreads each doc's signal thinner, which plausibly
  costs exact-phrase DIRECT_LOOKUP hits while slightly helping synthesis
  CONCEPTUAL queries. Per-query extremes: D_kwDOCZduT84Airbw 1.0 -> 0.0,
  D_kwDOCZduT84AWbw2 0.667 -> 1.0.
- `constructed-mh-03` still 0.0; unchanged lever remains reranking.
- Dev index currently holds the structural build; committed config default
  is still naive, so the next plain `rag-index build --force` restores it.
