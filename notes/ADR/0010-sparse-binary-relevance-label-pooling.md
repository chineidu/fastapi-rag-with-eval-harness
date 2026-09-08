---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Sparse binary relevance via top-30 semantic search pooling

## Context

Labeling all (query, doc) pairs is infeasible: 70 queries × 615
docs = 43,050 judgments for a solo engineer. The labeling pipeline
must dramatically reduce this without compromising the eval.

## Decision

Use **sparse binary relevance** with **pooling via semantic
search**:

1. For each query, semantic search retrieves the top-30 candidate
   docs.
2. The LLM judge labels each candidate as relevant or irrelevant
   given the query, reference answer, and full doc content.
3. Total judgments: 70 × 30 = **2,100** (vs. 43,050 for full
   labeling).
4. Most queries have 1-3 relevant docs (DIRECT_LOOKUP typically
   1; MULTI_HOP and CONCEPTUAL 2-5), producing ~140-210 total
   relevant labels.

The standard TREC/BEIR pooling assumption applies: any doc that
never surfaced in the top-30 semantic search is assumed
irrelevant. This biases recall@k slightly upward (some relevant
docs are missed) but the *relative* comparison (before vs. after
across runs) remains valid because both runs are evaluated against
the same ground truth.

## Alternatives considered

- **Complete labeling** (43,050 judgments): infeasible for a solo
  engineer; no marginal value given the pooling assumption.
- **Random sampling**: would miss most relevant docs (most
  queries have 1-3 relevant out of 615 — random sampling is
  hopelessly sparse).

## Consequences

Easier: 20x reduction in labeling cost (~2,100 judgments vs.
~43,050); the LLM judge can be tuned for high accuracy on the
top-30 candidates.

Harder: absolute recall@k numbers are slightly inflated; the
eval measures relative improvement (before vs. after), not
absolute retrieval quality vs. an oracle.

## Rationale

Pooling via semantic search is the standard approach in IR
evaluation (TREC, BEIR). The 30-candidate cutoff balances
coverage (most relevant docs surface in top-30) against cost
(more candidates = more LLM calls). The pooling assumption
introduces a small upward bias in absolute recall@k, but this
is the same bias for every run, so relative comparisons remain
valid — which is what the eval harness is for.
