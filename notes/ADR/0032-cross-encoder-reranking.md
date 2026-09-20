---
status: ratified
date: 2026-09-20
deciders: project owner
---

# Cross-encoder reranking over fused chunk candidates

## Context

Hybrid RRF is nearly neutral overall (OVERALL 0.594 vs 0.592 dense-only)
with MULTI_HOP +0.096 as the win. The flagged hard case
`constructed-mh-03` still scores 0.0 at k=10 while 6/7 judged docs rank
12-28 at k=30: the file-upload half saturates the top 10. Rank-only
fusion cannot fix token-level relevance inside the candidate window.

## Decision

Add a FastEmbed `TextCrossEncoder` reranker (`Xenova/ms-marco-MiniLM-L-6-v2`
default, apache-2.0, 0.08GB) as an optional stage in `LocalRetriever`.
Rerank the top-30 fused chunk hits before per-doc dedupe (ADR-0021),
then dedupe to k. Gated by `retriever_config.rerank_enabled`
(default false) so the dense-only and hybrid baselines reproduce.

How it works: unlike bi-encoders, which embed the query and each chunk
separately for fast cosine search, the cross-encoder feeds the query
and one chunk together through the transformer, so attention scores
every query-token to chunk-token interaction. It outputs one relevance
score per pair; the 30 candidates are sorted by that score and the top
hits go to dedupe and generation. Per-query cost is linear in the
candidate count, which is why it runs on 30 fused hits and never on the
full collection.

## Alternatives considered

- MiniLM-L-12-v2: slightly better quality at 0.12GB but slower; lost on speed for an educational loop.
- BAAI/bge-reranker-base (1.04GB, mit): best quality but heaviest download; lost on weight.
- Rerank after dedupe: cheaper (k docs) but loses chunk-level signal where saturation happens; lost on the hard case.

## Consequences

What becomes easier? Token-level relevance inside the candidate window
becomes measurable via `rag-eval diff`. Harder? Adds ONNX cross-encoder
download, per-query CPU cost, one more config surface.

## Rationale

Cross-encoders trade speed for accuracy on a small set, which is exactly
the 12-28 band the hard case lives in. Before-dedupe placement is the
only option that can promote those buried chunks. L-6-v2 keeps the loop
fast and matches the bge-small dense family.
