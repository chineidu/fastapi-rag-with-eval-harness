# Phase 2 roadmap - slices

Handoff notes for a fresh session. Slices are ordered by execution
priority: retrieval (done), generation (done, committed), API service incl. streaming (next: 3a then 3b), then distribution
at deploy time. Update statuses here as slices land.

Status legend: done / next / planned / deferred.

## Slice 1 - Hybrid retrieval (done, committed)

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
- Landed in `9f9df74` (verified 2026-09-15; the 2026-09-14
  "uncommitted / parallel session" note is stale).

## Slice 2 - Generation (done, committed)

The Phase 2 core deliverable: without generation there is no RAG
service, only a retriever. Chosen before distribution because it
blocks nothing downstream and carries the project's open design
questions.

Decisions landed (ADR-0027, ratified):

- Stack: OpenRouter + instructor, `rag_config.llm` as config source.
- `agenerate` is async-only over `list[SearchHit]`; amends the
  `RetrieverAdapter` protocol (ADR-0004) and `LocalRetriever`.
- Context: best-chunk texts in rank order, no truncation in v1.
- Prompt: few-shot grounded, Anthropic XML style with question last.
- Answer: `GeneratedAnswer` with light citation clamping; transport
  errors raise; answer eval deferred.

Verification: `ruff check` + `ty check` pass, `pytest -q` 445 passed.

## Slice 3 - API service (next, in two parts)

New `src/api/` router wrapping retrieve + generate into a service.
Depends on slice 2 (landed). No distribution dependency for local/dev use.
Streaming is bundled here as a second part, not a new slice, since it
shares the same service wiring behind a separate endpoint. Each part
lands in 2-3 reviewable commits, never one big commit.

### Slice 3a - Non-streaming service (first)

- Contract + ADR first: route shape, request/response schemas, error
  mapping for raised transport errors.
- Commits: (1) request/response schemas; (2) route + service wiring
  (`src/api/` router, app factory, health endpoint); (3) error mapping
  + TestClient tests (faked generator, no network).

### Slice 3b - Streaming generation (second)

- No new libraries: raw `openai.AsyncOpenAI` SSE (`stream=True`) or
  `instructor.Partial[GeneratedAnswer]`; promote `openai` to a direct
  dep via `uv add openai`.
- Contract + ADR first: `astream` shape (deltas vs snapshots), SSE event
  format (token events + terminal metadata with citations/grounded),
  mid-stream error semantics, another ADR-0004 protocol amendment.
- Commits: (1) `astream` on `RAGGenerator` + unit tests with a faked
  async-iterator client; (2) SSE endpoint + event format + streaming
  tests. The non-streaming path stays for the eval harness.

## Slice 4 - Index distribution (deferred, deploy-time)

Deferred to a future ADR (draft as `proposed` first). Full workflow draft
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
- `notes/ADR/0025-hybrid-search.md`, `notes/ADR/0027-generation-with-citations.md`, `notes/ADR/OVERVIEW.md`
- `README.md` section 4.4 - hybrid retrieval architecture
- Eval history: `data/.rag-eval/runs.db` (baseline run 2, final run 13)
