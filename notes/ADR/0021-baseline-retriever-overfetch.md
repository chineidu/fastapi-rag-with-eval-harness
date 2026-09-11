---
status: proposed
date: 2026-09-11
deciders: project owner
---

# Baseline retriever: chunk search with doc-level over-fetch

## Context

The harness asks the adapter for `k` documents (ADR-0004, ADR-0012), but the
vector store returns chunks and a single document can own many chunks.
Searching for `k` chunks and deduplicating to documents therefore returns
fewer than `k` documents whenever the top chunks cluster in a few files.
`recall@k` is then computed over a short candidate list, so the metric
measures chunk clustering rather than document retrieval quality.

## Decision

`LocalRetriever.retrieve(query, k)` over-fetches chunks before
deduplicating:

1. Embed the query with the configured embedder (the same model used for
   indexing).
2. Search the vector store for `k * overfetch_factor` chunks.
3. Walk the hits in score order and keep the first (best) hit per
   `doc_path`.
4. Truncate to the `k` best documents and return a `RetrievalResult`.

`overfetch_factor` is an integer `>= 1`, default `5`, configured via
`retriever_config.overfetch_factor` in the app config. It is the multiplier
applied to `k` to size the chunk-level candidate window so the dedupe step
still has enough material to fill `k` document slots.

A document is represented exactly once in the result, regardless of how many
of its chunks fall inside the window. After the first (best) hit for a
`doc_path` is kept, every later hit for the same path is skipped, so chunk
volume cannot buy a document extra slots or inflate its score: a document's
rank is the score of its single best chunk. A document that contributes 20 of
the 50 fetched chunks still occupies exactly one of the `k` result slots, and
its 19 remaining chunks are discarded.

What the factor does, by example (`k=10`):

- `overfetch_factor=5` fetches 50 chunks. If 20 come from one document and
  the other 30 span 12 documents, dedupe yields at most 13 candidates (the
  large document once) and the result is the best 10.
- `overfetch_factor=1` fetches 10 chunks. If 6 of them come from one
  document, the adapter can return as few as 4 documents, and a relevant
  document whose best chunk ranked 11th or lower is unreachable.

The over-fetch is one wider query. It adds no re-ranking and does not change
the relative order of documents already inside the window.

## Alternatives considered

- **No over-fetch (`overfetch_factor=1`)**: matches "top-k cosine, dedupe to
  docs" literally, but structurally caps doc recall at the number of distinct
  documents present in the top `k` chunks, so it loses on metric validity.
- **Adaptive fetch**: repeat queries with a growing window until `k` unique
  documents or the index is exhausted. Removes the magic constant, but costs
  multiple round trips and more state for a baseline.
- **Fixed candidate pool** (for example always 100 chunks): simple, but
  decouples the window from `k`, so small `k` pays a large, unexplained cost.
- **Chunk-level ground truth**: removes the mismatch entirely, but invalidates
  the existing doc-level ground truth and contradicts ADR-0012.

## Consequences

Easier: doc-level recall reflects document ranking rather than chunk
clumping; the factor is tunable per experiment without code changes.

Harder: one extra config knob; the factor is a heuristic, and a single huge
document can still crowd out the window (only adaptive fetch would fully fix
that). Over-fetching transfers and scores more candidates per query, which is
negligible at this corpus size but not free.

## Rationale

The harness contract is doc-level while the index is chunk-level; something
must bridge the two. Over-fetching is the smallest change that makes
`recall@k` measure what it claims to measure, and a configurable multiplier
keeps the cost explicit and experiment-tunable.
