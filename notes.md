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
