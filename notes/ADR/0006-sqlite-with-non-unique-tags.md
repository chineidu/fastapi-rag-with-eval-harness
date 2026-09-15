---
status: ratified
date: 2026-08-02
deciders: project owner
amended_by: 0026-unique-run-tags
---

# SQLite for run history with non-unique tags

## Context

Eval runs accumulate over time. The harness needs to query across
runs ("show me all MULTI_HOP scores from the last 30 days"), diff
two runs, and persist results cheaply.

## Decision

Use **SQLite** (default path `data/.rag-eval/runs.db`, gitignored)
with this schema (abbreviated):

- `runs` table: `id`, `tag`, `created_at`, `status` ('running' |
  'complete' | 'partial'), `config` (JSON), `metadata` (JSON).
- `query_results` table: links to `runs`, stores `query_id`,
  `category`, `k_value`, `retrieved_docs` (full ranked list as
  JSON), `ground_truth`, `recall_at_k`, `precision_at_k`,
  `latency_ms`, `status`, `error`, `metrics_extra` (JSON).

**Tags are non-unique.** Multiple runs can share a tag; `diff`
resolves to the latest run with that tag. Use `--run-id` for
precise comparison.

## Alternatives considered

- **JSON-per-run files**: works for one-off diffs but requires a
  directory scan and manual parsing for cross-run queries.
- **Unique tags**: forces the user to invent new tag names for
  every run; makes variance experiments awkward.

## Consequences

Easier: cross-run queries are SQL; single file, no setup; small
(~70 rows per run); tags describe *what kind of run* this is, not
*which specific run* this is.

Harder: SQLite adds a dependency (stdlib only, no install
required); the `retrieved_docs` column stores the full ranked
list, which is larger than just the metric but enables per-query
drill-down on regressions.

## Rationale

SQLite enables the query patterns the harness needs ("show me
all MULTI_HOP scores from the last 30 days"). Non-unique tags
follow the pattern used by MLflow and W&B — tags describe the
experiment type, not the specific run, so you can run the same
config 3 times to measure variance under one tag. Storing the
full `retrieved_docs` list (not just the metric) costs a bit of
disk but enables per-query regression debugging.
