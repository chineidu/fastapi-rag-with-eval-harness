---
status: ratified
date: 2026-09-17
deciders: project owner
---

# Structural chunker behind a strategy flag

## Context

Naive fixed-size slicing (ADR-0018) cuts mid-sentence and mid-code. Plan Phase 3 calls for header-aware v2 measured against the naive baseline via the doc-level harness (ADR-0012).

## Decision

Add a structural strategy selected by `indexer_config.chunk_strategy` (default naive): markdown packs ATX-all-levels plus Setext sections skipping fenced code; Python packs top-level def/class blocks via `ast`; either overflow path char-slices that unit only; directory dispatch by suffix; adapter reports the strategy in metadata.

## Alternatives considered

- Recursive headers-paragraphs-sentences: finer boundaries but more code and tuning, so it lost.
- Replace naive outright: simpler diff but destroys A/B comparison, so additive flag won.
- Blank-line / body-aware overflow: nicer cuts but more paths; char-slice fallback won for simplicity.

## Consequences

Easier: boundary-respecting chunks, baseline preserved for diff, no new deps. Harder: fingerprint changes force full reindex on switch; overlap only applies inside fallback slices.

## Rationale

Captures most of the structural gain (headers travel with bodies, methods stay whole in the common case) with minimal new code; overflow nicety deferred until eval justifies it.
