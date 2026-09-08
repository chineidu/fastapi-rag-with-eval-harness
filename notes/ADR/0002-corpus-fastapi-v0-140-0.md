---
status: ratified
date: 2026-08-02
deciders: project owner
---

# FastAPI v0.140.0 docs as corpus

## Context

The eval harness needs a concrete corpus to evaluate retrieval
against. The corpus must be stable, reproducible, and rich enough to
exercise a range of retrieval challenges.

## Decision

Use **615 FastAPI documentation files** from the upstream repo at
v0.140.0 (commit `255b912`, 2026-07-24, verified 2026-08-02):

- 154 EN markdown pages under `docs/en/docs/`
- 461 Python examples under `docs_src/`

Non-English translations are excluded by design.

## Alternatives considered

- **Different framework** (Django, Flask, Starlette): would test
  transferability but adds setup burden; FastAPI is the project's
  target library.
- **Latest commit instead of snapshot**: corpus would drift between
  eval runs, breaking reproducibility.

## Consequences

Easier: corpus is reproducible across runs; English-only markdown
keeps the text-cleaning pipeline simple.

Harder: corpus becomes out-of-date as FastAPI evolves. Snapshot
must be re-verified if the eval is rerun against a new FastAPI
version.

## Rationale

FastAPI is the project's target library, so the corpus is
domain-relevant. Pinning a snapshot (rather than HEAD) is required
for reproducibility — if the corpus changes between runs,
before/after diffs become meaningless. Non-English translations are
excluded to keep the text-cleaning pipeline simple; if multilingual
eval becomes a goal later, the corpus structure already supports it.
