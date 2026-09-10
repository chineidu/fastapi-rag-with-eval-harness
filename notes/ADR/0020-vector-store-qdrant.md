---
status: proposed
date: 2026-09-08
deciders: project owner
---

# Qdrant as the vector store (pluggable backend)

## Context

The indexer needs durable vector storage with similarity search. The
corpus is small (FastAPI docs, hundreds of chunks) but the project wants
real retrieval eval, not brute force. Self-hosted Qdrant via Docker is the
chosen backend. The backend must be swappable via config, not code edits,
so future backends (pgvector, LanceDB) are a one-file + one-config change.

## Decision

Use self-hosted Qdrant (Docker) as the default vector store backend. The
vector store is a package: a backend-agnostic `VectorStore` protocol, a
`BaseVectorStore` that owns the idempotency policy (model_id/dim check,
rebuild on mismatch), a `QdrantVectorStore` adapter, and a
`get_vector_store(cfg)` factory. The backend is selected by
`IndexerConfig.backend` (a `VectorStoreBackendEnum`); the factory maps it
to the adapter and builds it from its config slice. Each point carries
chunk metadata (chunk_id, doc_path, chunk_index, text) so the retriever
can map results back to documents. The collection records model_id and dim
(in a sentinel meta point) so retrieval never compares vectors from
mismatched models.

## Alternatives considered

- SQLite + manual / sqlite-vec / FAISS / numpy: all local, no service,
  but a real vector DB was chosen for eval realism.
- Managed Qdrant cloud: rejected for self-hosting to avoid API keys and
  network egress in dev.
- Protocol-only, no base class: simpler, but each backend re-implements
  idempotency and meta storage; the shared base avoids that.
- Pydantic discriminated union for backend config: heavier than sibling
  config fields for a 1-2 backend system.

## Consequences

Easier: native ANN search; config-driven backend swap; new backend is one
adapter file + one config field + one enum value + one factory entry.
Harder: adds qdrant-client dep and a Docker dependency for local dev;
connection failures are raised as clear errors; an extra base-class layer
to maintain.

## Rationale

A real vector store makes retrieval eval meaningful. Self-hosting keeps it
offline and free; the protocol + base + factory keep the rest of the
pipeline agnostic to the backend and make swaps config-driven.
