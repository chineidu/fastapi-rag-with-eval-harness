---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Error classification with exponential backoff retries

## Context

Adapter calls fail in two distinct ways: transient errors that
will likely succeed on retry (network blips, rate limits), and
deterministic errors that will fail every time (TypeError, bad
input). Treating them the same either wastes time retrying or
poisons the run with a single failure.

## Decision

Two error categories:

| Class | Examples | Behaviour |
|---|---|---|
| **Recoverable** | `ConnectionError`, `TimeoutError`, HTTP 5xx | Retry up to 2 times with exponential backoff |
| **Fatal** | `ValueError`, `TypeError`, malformed results | Log + skip, mark query as `"error"` |

Reporting:

- Failed queries are counted: `"✓ 67/70 completed, ✗ 3
  failures"`.
- Incomplete runs get status `"partial"` (not `"complete"`).
- `diff` warns if either run is incomplete.

## Alternatives considered

- **Retry everything**: wastes time on fatal errors (loops until
  max retries).
- **Retry nothing**: fragile; one network blip poisons the run.
- **Configurable per-error-type retry policy**: flexible but
  adds config surface for a tool that runs locally.

## Consequences

Easier: transient errors are handled gracefully; deterministic
errors don't waste retry budget; the user sees clear success/
failure counts.

Harder: the adapter author must distinguish recoverable from
fatal errors (or trust Python's exception hierarchy).

## Rationale

A transient network error will likely succeed on retry; a type
error in the adapter will fail every time. The two-class scheme
matches reality without over-engineering. Exponential backoff
avoids thundering-herd on rate-limited services and is standard
practice for transient failures.
