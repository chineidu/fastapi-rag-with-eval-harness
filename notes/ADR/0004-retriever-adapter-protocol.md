---
status: ratified
date: 2026-08-02
deciders: project owner
amended_by:
  - 0019-retrieved-document-type
  - 0027-generation-with-citations
  - 0029-streaming-generation
---

# RetrieverAdapter protocol

## Context

The adapter contract defines what the harness can call and what it
expects back. The contract shapes the harness's ability to evaluate
retrieval and (later) generation independently.

## Decision

```python
@dataclass
class RetrievalResult:
    documents: list[tuple[str, float]]  # [(doc_path, score), ...]
    metadata: dict[str, object]  # chunking info, model name, config hash, etc.


class RetrieverAdapter(Protocol):
    def retrieve(self, query: str, k: int = 10) -> RetrievalResult: ...
    def generate(self, query: str, documents: list[str]) -> str: ...  # deferred
```

Three rules follow from this contract:

1. `retrieve()` and `generate()` are separate methods; the harness
   calls them independently.
2. The harness owns timing measurement (wraps adapter calls with
   `time.monotonic()`).
3. The adapter abstracts chunking — the harness operates at the
   document level; chunk-to-doc deduplication is the adapter's
   responsibility; chunk metadata goes in `metadata` for
   auditability.

## Alternatives considered

- **Single `query()` method** that returns both documents and a
  generated answer: couples retrieval eval to generation cost.
- **Adapter self-reports timing**: measurements may differ in
  methodology or be omitted; harness wrapping is more consistent.
- **Harness operates at chunk level**: makes chunking part of the
  contract, so chunking-strategy changes break ground truth.

## Consequences

Easier: retrieval eval works without triggering LLM calls; the
adapter contract works for retrieval-only RAG systems with no
generator; chunking changes don't invalidate ground truth; timing
is consistent across adapters.

Harder: adapters must implement chunk-to-doc deduplication; chunk
metadata discipline is on the adapter author.

## Rationale

Separating retrieve and generate means the harness can call
`retrieve()` for retrieval eval today and add `generate()` for
generation eval later without changing the retrieval path.
Harness-owned timing ensures every adapter is timed the same way.
Doc-level ground truth keeps the eval stable when chunking
strategy changes — chunking is an implementation detail of the RAG
system, not an eval concern.
