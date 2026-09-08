---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Per-category metric aggregation

## Context

The eval produces per-query metrics (recall@k and precision@k
for each of 70 queries). How should these be aggregated to give
an overall picture?

## Decision

Two-level aggregation:

1. **Per-category mean** — mean recall@k and precision@k across
   all queries in each category (`DIRECT_LOOKUP`, `MULTI_HOP`,
   `CONCEPTUAL`).
2. **Overall mean** — mean across all 70 queries (unweighted).

`diff` reports both per-category and overall deltas. The
per-category breakdown tells you *where* the system is weak and
*what kind* of change helped; the overall mean answers "is the
system better on average?".

Unanswerable queries (see ADR-0009) are excluded from recall@k
means and reported separately as an `unanswerable` count.

## Alternatives considered

- **Overall mean only**: hides category-level regressions (e.g.,
  CONCEPTUAL drops 0.10 while DIRECT_LOOKUP improves 0.05; the
  average looks flat but you've broken conceptual retrieval).
- **Weighted aggregation** (by category size): the 38/17/15
  split already makes the unweighted mean roughly
  category-weighted; explicit weighting adds complexity for
  little gain.

## Consequences

Easier: per-category deltas pinpoint regressions; overall mean
gives a single number for tracking.

Harder: report formatting must show both per-category and
overall; the user has to look at the right level when
interpreting results.

## Rationale

An overall metric hides regressions in specific categories. If
CONCEPTUAL drops 0.10 but DIRECT_LOOKUP improves 0.05, the
average looks flat — but you've broken conceptual retrieval.
Per-category breakdown tells you *where* the system is weak and
*what kind* of change helped. The delta answers the before/
after question directly. Thresholds (absolute + relative)
filter out noise from meaningful change.
