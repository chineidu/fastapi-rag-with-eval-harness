---
status: ratified
date: 2026-09-08
deciders: project owner
amends: 0004-retriever-adapter-protocol
---

# RetrievedDocument element type

## Context

Amends ADR-0004 (retriever adapter protocol) for the element shape
only. RetrievalResult.documents and QueryOutcome.retrieved_docs move from
list[tuple[str, float]] to list[RetrievedDocument].

## Decision

Introduce frozen RetrievedDocument(doc_path: str, score: float) with int
coerced to float, use list[RetrievedDocument] in both dataclasses, persist
as [{"doc_path": ..., "score": ...}] JSON in SQLite.

## Alternatives considered

- NamedTuple with same fields: keeps unpacking but splits style.
- Keep tuples: zero churn but weak typing.

## Consequences

Easier: attribute access, single coercion point. Harder: breaks unpacking
in runner, store, tests, README; old DB JSON shape unreadable (accepted:
DB not live).

## Rationale

Attribute access beats index access for a public adapter contract, and
matching Python and JSON shapes avoids a second representation.
