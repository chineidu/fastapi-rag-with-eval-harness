---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Harness internal module split

## Context

The harness has multiple responsibilities: orchestration, metrics
computation, persistence, and config. A single `harness.py` would
be unmaintainable. The internal structure should make the
boundaries between these concerns clear.

## Decision

Split the harness into four internal modules plus three shared
modules:

- `EvalRunner` — orchestrates queries, concurrency, error
  handling.
- `MetricsCalculator` — pure functions: `recall_at_k`,
  `precision_at_k`, `aggregate_by_category`.
- `ResultStore` — SQLite CRUD: `create_run`, `save_result`,
  `finalize_run`, query for diff.
- `GroundTruthLoader` — loads and validates ground truth JSONL.

Plus:

- `adapter.py` — `RetrieverAdapter` protocol + `RetrievalResult`
  dataclass (see ADR-0004).
- `config.py` — OmegaConf structured config (see ADR-0016).
- `cli.py` — Typer CLI entry point.

## Alternatives considered

- **Single `harness.py`**: too much code in one file; hard to
  test in isolation.
- **More granular split** (e.g., separate `runner/` package):
  premature for a 70-query tool; over-engineering.

## Consequences

Easier: each module is testable in isolation; pure functions
(`MetricsCalculator`) are easy to verify; persistence is
encapsulated in `ResultStore`.

Harder: cross-module imports need a clear hierarchy (CLI →
runner → adapter/store/metrics/loader).

## Rationale

Four modules map cleanly to the four core concerns: orchestration,
computation, persistence, input. The split is large enough to
prevent monolithic code but small enough that the entire harness
is still readable in one sitting.
