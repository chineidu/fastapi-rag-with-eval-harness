---
status: ratified
date: 2026-08-02
deciders: project owner
---

# Chunking is an adapter-internal concern

## Context

The eval harness operates at the document level: it asks "which
documents were relevant?" and "which documents were retrieved?"
But RAG systems typically chunk documents before indexing. Where
does chunking fit in the contract?

## Decision

The adapter **abstracts chunking away from the harness**. The
harness:

- Asks for documents (`adapter.retrieve()` returns a list of
  `(doc_path, score)` pairs).
- Knows nothing about chunks, embeddings, or chunk-to-doc
  mapping.

The adapter is responsible for:

- Chunking strategy choice.
- Chunk-to-doc deduplication when reporting retrieved docs.
- Including chunk metadata (chunk counts, overlap settings,
  embedding model, config hash) in `RetrievalResult.metadata`
  for auditability.

This keeps the harness simple and the ground truth stable: doc
identifiers (see ADR-0011) don't change when chunking strategy
changes.

## Alternatives considered

- **Harness operates at chunk level**: makes chunking part of
  the contract; ground truth becomes chunk-level;
  chunking-strategy changes invalidate ground truth.
- **Both chunk and doc levels exposed**: doubles the API
  surface for marginal value.

## Consequences

Easier: harness is simpler; ground truth is stable across
chunking experiments (changing chunk size, overlap, or strategy
doesn't require re-labeling); adapter has full control of its
chunking strategy.

Harder: chunk-level debugging requires inspecting
`RetrievalResult.metadata`; the adapter must implement
chunk-to-doc deduplication.

## Rationale

Chunking is an implementation detail of the RAG system, not an
eval concern. The harness asks about documents because that's
what ground truth labels — putting chunking in the contract
would make every chunking experiment require re-labeling.
Documenting the chunking config in `metadata` preserves the
audit trail without making it part of the contract.
