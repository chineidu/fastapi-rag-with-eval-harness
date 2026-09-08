---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Unanswerable queries marked answerable: false

## Context

Real user questions are not always answerable from the corpus
(bug reports, version-specific issues, regressions in older
versions). The harness needs to handle these without poisoning
recall@k with phantom zeros.

## Decision

When the LLM judge finds no relevant doc among the top-30
candidates for a query, the record is written with:

```json
{
  "query_id": "...",
  "answerable": false,
  "relevant_docs": [],
  ...
}
```

No synthetic fallback doc is selected. The harness computes
recall@k means over **answerable queries only** and reports the
unanswerable count separately in `RunSummary`. A future
generation-eval can use these queries to test abstention (the
system should say "not in my knowledge base" instead of
hallucinating).

A schema validator enforces consistency: `answerable: true`
requires ≥1 `relevant_docs`; `answerable: false` requires empty
list.

## Alternatives considered

- **Synthetic fallback to top-1 retrieved doc**: would mark
  unanswerable queries as having a "relevant" doc, inflating
  recall@k and hiding real failure modes.
- **Drop unanswerable queries entirely**: loses the signal; we
  want to know what fraction of real user questions the corpus
  can't answer.

## Consequences

Easier: recall@k reflects actual retrieval quality; the
unanswerable count is a separate signal about corpus coverage.

Harder: harness code must handle empty `relevant_docs` lists
consistently (loader, runner, diff command); schema validator
must enforce consistency.

## Rationale

Treating unanswerable queries as having a synthetic "best guess"
relevant doc would corrupt recall@k — the system would get
"credit" for retrieving a doc it never actually answered with.
Marking them explicitly preserves the signal (corpus coverage)
without polluting retrieval quality metrics. The abstention use
case (test that the system says "I don't know" instead of
hallucinating) is the natural future application.
