# Phase 2 roadmap - slices

Handoff notes for a fresh session. Slices are ordered by execution
priority: retrieval (done), generation, API service, then distribution
at deploy time. Update statuses here as slices land.

Status legend: done / next / planned / deferred.

## Slice 1 - Hybrid retrieval (done, uncommitted)

Persisted tantivy BM25 index alongside Qdrant dense, fused at query
time with RRF. `hybrid_enabled: true` is the committed config.

- Code: `src/app/hybrid.py` (`TantivyIndex`, `rrf_fuse`), wiring in
  `src/app/adapter.py`, `src/app/indexer.py`, `src/app/cli.py`;
  config in `src/config/config.yaml` -> `retriever_config`.
- Tuned config: `sparse_k=50`, `rrf_k=30`,
  `dense_weight=0.75` / `sparse_weight=0.25` (normalized ratio),
  Snowball stopwords, min token length 3.
- Eval (recall@10): OVERALL 0.594 vs 0.592 dense-only; MULTI_HOP
  +0.096 (+19.8%); DIRECT_LOOKUP -0.042, CONCEPTUAL -0.049.
- ADR 0025 (ratified). Tuning sweep details in `notes.md`.
- As of 2026-09-14 the whole slice is uncommitted and a parallel
  session may be committing it. Check `git log` / `git status` first;
  do not redo the work.

## Slice 2 - Generation (next)

The Phase 2 core deliverable: without generation there is no RAG
service, only a retriever. Chosen before distribution because it
blocks nothing downstream and carries the project's open design
questions.

Decisions locked earlier:

- Stack: OpenRouter + instructor.
- `generate()` becomes async-only (`agenerate`); amend the
  `RetrieverAdapter` protocol (ADR-0004 amendment).
- Context assembled from retrieved `RetrievedDocument`s.

To design first (contract + ADR before implementation): prompt shape,
grounding and citation format, and whether answer eval lands now or
later (ADR-0004 currently defers generation eval).

## Slice 3 - API service (planned)

New `src/api/` router wrapping retrieve + generate into a service.
Depends on slice 2. No distribution dependency for local/dev use.

## Slice 4 - Index distribution (deferred, deploy-time)

Deferred to ADR 0026 (draft as `proposed` first). Full workflow draft
in `notes.md` -> "Hybrid index workflow - option 2".

Shape: CI builds image v123 once; a one-off job builds the Qdrant
collection `docs_v123` and pushes `tantivy-v123.tar.gz` plus
`manifest-v123.json` to S3; replicas boot with `INDEX_VERSION=v123`,
fetch the manifest, download and verify the tarball fingerprint, and
fail readiness on mismatch. Cleanup keeps the last 1-2 versions;
rollback is redeploy with the old version.

Priority: independent of slices 2-3. Not urgent until a real
multi-replica deploy is on the table; then it can run in parallel.

## Parked items (flagged, not fixed)

- `rag-index build` skip path: if Qdrant is up to date but the
  tantivy index is missing, the build is skipped and hybrid silently
  degrades to dense-only. Workaround: `--force`.
- ADR 0025 Consequences references `rag-eval diff v1-baseline v2-hybrid`;
  the actual tag is `baseline`.
- Qdrant container running v1.15.1 vs `docker-compose.yml` pin v1.16.3.

## Key references

- `notes.md` - iteration log, tuning sweep, distribution workflow draft
- `notes/ADR/0025-hybrid-search.md`, `notes/ADR/OVERVIEW.md`
- `README.md` section 4.4 - hybrid retrieval architecture
- Eval history: `data/.rag-eval/runs.db` (baseline run 2, final run 13)
