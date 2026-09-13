---
status: ratified
date: 2026-08-02
deciders: project owner
amended_by: 0024-constructed-multi-hop-backfill
---

# 70 evaluation queries from GitHub discussions and StackOverflow

## Context

The eval set needs real user questions to be meaningful. Synthetic
questions are too easy and don't reflect how real users phrase
things.

## Decision

Use **70 real user questions** from two sources:

- 40 GitHub discussion threads (`fastapi/fastapi` Discussions,
  Questions category).
- 30 StackOverflow Q&A threads (tag `fastapi`, sorted by votes,
  favoring accepted answers).

Each tagged with one of three difficulty categories:

- `DIRECT_LOOKUP` (38): answer lives in a single doc page.
- `MULTI_HOP` (17): answer requires synthesising across 2+ pages.
- `CONCEPTUAL` (15): requires understanding concepts spread across
  multiple pages.

## Alternatives considered

- **Synthetic questions** (engineer writes them): too easy;
  hindsight bias; doesn't reflect real user phrasing.
- **GitHub only**: smaller pool; biases toward open-source user
  style.
- **Larger eval set (200+)**: more statistical power but ~3x the
  labeling cost; 70 fits the 70×30=2,100 judgment budget.

## Consequences

Easier: eval reflects real user phrasing and difficulty;
categories enable breakdown by query type.

Harder: GitHub and SO questions are sometimes version-specific,
unanswerable from the current docs, or about non-doc concerns
(bug reports, regressions). Unanswerable cases need explicit
handling (see ADR-0009).

## Rationale

Real user questions test retrieval against the distribution of
phrasing the system will see in production. The three categories
cover the spectrum from literal lookup to synthesis, which is
what matters for a RAG system. The 70-question size is a pragmatic
tradeoff: large enough to see category-level signal, small enough
to label via the LLM-judge + top-30 pooling pipeline (see
ADR-0010).
