---
status: proposed
date: 2026-09-15
deciders: project owner
amends: 0006-sqlite-with-non-unique-tags
---

# Unique run tags with timestamp suffixes

## Context

ADR-0006 keeps run tags non-unique by design: repeated runs share a
label and `diff` resolves the label to the latest run. In practice
this shadows runs. In the current database runs 3 and 4 both carry
`v2-hybrid`, and the partial run 3 (6/83) is unreachable by tag
because resolution always lands on the later run; `list` shows labels
that cannot be told apart without falling back to `--run-id`.

## Decision

When `rag-eval run` receives a tag that already exists (exact match,
including the generated `auto-` tags), append a UTC `-HHMMSS` suffix
before inserting; if that too is taken (same-second collision), append
`-2`, `-3`, and so on until the tag is free. The first run under a tag
keeps the plain name, so `baseline` stays a stable pointer to the
original baseline. The rule lives in `ResultStore.unique_tag()` and is
applied by the CLI `run` flow; the `runs` schema, `create_run`, and
`resolve_ref` semantics are unchanged, and existing rows are not
migrated.

## Alternatives considered

- Random four-character suffix (`baseline-k7m2`): collision-proof
  without coordination, but non-deterministic and opaque; lost on
  auditability, since this harness favours predictable identifiers.
- Numeric counter (`baseline-2`): shortest and fully predictable, but
  needs a max-suffix scan before insert and says nothing about when
  the run happened.
- Keep non-unique tags and rely on `--run-id`: rejected because it is
  exactly the present papercut. The tag is what humans type; duplicate
  labels make it ambiguous, and the id is an implementation detail.

## Consequences

Easier: every run has a human-typable unique handle; `list` and `diff`
cannot silently pick the wrong run; suffixes are chronological by
construction and sortable. Harder: the plain tag now denotes the first
run, not the latest (a deliberate change from ADR-0006's latest-wins
resolution), so a workflow that re-ran under one tag must switch to
prefix grouping (`baseline%`). Historical duplicate rows keep their
labels; no migration is performed.

## Rationale

Timestamp won over random characters because this project is built
around reproducible comparisons: a suffix you can predict, sort, and
read as a time is more useful than an opaque token, and the suffix is
only applied on collision, so the common case (`--tag chunk-512`,
first use) stays clean. `--run-id` remains the precise escape hatch.
