---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Harness and adapter separation

## Context

The eval harness needs to be reusable across RAG projects (this one
and future ones). If the harness is coupled to one codebase,
porting it becomes a rewrite instead of a copy.

## Decision

Split into two components:

- **Harness** (`src/eval_harness/`): runner, metrics engine, CLI,
  SQLite persistence, diff command. Knows how to evaluate any
  adapter that follows the contract.
- **Adapter** (project-specific): bridges the harness to a specific
  RAG codebase. Implements `RetrieverAdapter`. Written once per
  project.

The harness module is designed to be copyable to future repos; the
adapter is the only per-project code.

## Alternatives considered

- **Single monolithic tool**: easier to write first, but couples
  the harness to one codebase and makes porting a rewrite.
- **Plugin system with discovery**: more flexible, but adds a
  dependency-injection layer for a tool that only has one consumer
  per repo.

## Consequences

Easier: the harness is portable across projects; the adapter
boundary is the single decision that makes the tool reusable.

Harder: requires discipline to keep RAG-specific logic out of the
harness; tests need to verify the harness works without a real
adapter (dummy/oracle adapters).

## Rationale

The adapter pattern is a standard design pattern for inversion of
control. Without this boundary, the harness would be coupled to one
RAG codebase and unportable. The boundary also enables testing the
harness itself with a dummy adapter that returns known results
(used in B9 acceptance).
