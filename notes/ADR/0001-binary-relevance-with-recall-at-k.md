---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Binary relevance with recall@k as starting metric

## Context

The harness needs a relevance scheme and a primary metric to evaluate
retrieval quality. Many options exist: graded vs binary relevance;
recall@k vs precision@k vs MRR vs NDCG. The choice affects labeling
cost, metric interpretability, and how easily other metrics can be
added later.

## Decision

Use **binary relevance** (each document is relevant or not, no
intermediate scores) and **recall@k** as the primary retrieval metric.

## Alternatives considered

- **Graded relevance** (essential / somewhat useful): more
  information per doc, but 5-10x the labeling cost.
- **MRR or NDCG as primary**: more sensitive to ranking, but
  requires graded relevance labels.

## Consequences

Easier: labeling is faster; recall@k is interpretable; no LLM
needed for retrieval eval. Other metrics (precision@k, MRR, NDCG)
are derivable from the same per-query data later.

Harder: cannot distinguish "essential" from "somewhat useful" docs.
Unanswerable queries (where no doc in the corpus answers) require
explicit handling (see ADR-0009).

## Rationale

Recall@k is the most fundamental retrieval question — "did we find
the right docs?" — and is standard in information retrieval. Binary
relevance is the simplest labeling scheme that produces useful
signals. Starting with the simplest meaningful metric avoids
premature complexity; other metrics can be derived incrementally
without re-labeling.
