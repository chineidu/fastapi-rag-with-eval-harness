---
status: ratified
date: 2026-09-13
deciders: project owner
amends: 0007-eval-queries-github-stackoverflow
---

# 13 constructed MULTI_HOP queries to backfill answerable set

## Context

Baseline run 6 covers 70 queries with 31 unanswerable and 39 answerable
(24 DIRECT_LOOKUP + 4 MULTI_HOP + 11 CONCEPTUAL). MULTI_HOP recall@10 of
0.479 on n=4 is too noisy for iteration. Real-user MULTI_HOP questions are
mostly bug reports unanswerable from docs.

## Decision

Append 13 LLM-drafted, doc-pair-bounded MULTI_HOP queries
(`constructed-mh-01` through `constructed-mh-13`, `source: constructed`,
`answerable: true`, at least two `relevant_docs`) to
`data/ground_truth.jsonl` (70 records growing to 83). Keep all 31
unanswerable records for abstention and hallucination eval.

## Alternatives considered

- Synthetic pairs without judge: guarantees at least two docs but skips
  the top-30 plus judge validation, and drifts from the ADR-0010 pipeline.
- Replace unanswerables: holds the total at 70 but destroys the
  abstention signal kept per ADR-0009.

## Consequences

What becomes easier: MULTI_HOP answerable grows from 4 to 17, giving
stable per-category means.

What becomes harder: the overall mean shifts, so old and new runs are
only comparable via per-category `diff`. The `source: constructed` value
breaks the github-or-stackoverflow assumption in ADR-0008.

## Rationale

Pair-bounded drafting guarantees synthesis across two or more pages, and
reusing the top-30 plus judge pipeline keeps labels consistent with the
existing 70 records.
