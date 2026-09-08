---
status: ratified
date: 2026-08-02
deciders: project owner
---

# ThreadPoolExecutor with bounded concurrency

## Context

The eval runner executes `adapter.retrieve()` for each of ~70
queries. The concurrency model affects throughput, adapter safety,
and how easy the harness is to debug.

## Decision

- Use `concurrent.futures.ThreadPoolExecutor` with a bounded pool
  size (default: 3, configurable via `.rag-eval.yaml` or
  `--concurrency`).
- Each adapter call is individually timed (the harness wraps each
  call with `time.monotonic()`).

## Alternatives considered

- **`asyncio`**: works for async adapters but adds an event loop
  for a tool that runs 70 queries — not a high-throughput server.
- **Unbounded concurrency**: can overwhelm the adapter (e.g.,
  re-loading an embedding model per call, hitting API rate
  limits).
- **Sequential execution**: simple but slow for 70 queries against
  a network-call adapter.

## Consequences

Easier: works with both sync and async adapters
(`asyncio.to_thread()` is a thin wrapper); threads are simpler to
reason about and debug than an async event loop; bounded
concurrency protects adapters from overload.

Harder: thread-safety is on the adapter author; per-call timing
has to account for thread overhead.

## Rationale

Threads are simpler than asyncio for a 70-query batch tool.
Bounded concurrency (default 3) balances speed and safety —
sequential is too slow, unbounded is dangerous. Individual call
timing remains clean since each call is independently measured.
Adapters that need different concurrency can configure it via
`.rag-eval.yaml`.
