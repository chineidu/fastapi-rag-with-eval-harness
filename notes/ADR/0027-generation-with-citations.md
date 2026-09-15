---
status: ratified
date: 2026-09-15
deciders: project owner
amends: 0004-retriever-adapter-protocol
---

# 0027 Few-shot grounded generation with structured citations

## Context

Slice 1 landed hybrid retrieval. Slice 2 needs generation to make a RAG
service. Prompt shape, citation format, context source, answer model,
module placement, and eval timing are open. ADR-0004 defers generation
eval and defines a sync `generate()` returning `str`.

## Decision

Few-shot grounded prompt over best-chunk `SearchHit` texts via OpenRouter
+ instructor, returning `GeneratedAnswer` with `answer`, `citations`,
`model_id`, and `grounded` flag. Protocol becomes async-only `agenerate`
taking `list[SearchHit]`. Prompt pair lives as `GenerationPrompt` in
`src/schemas/generation.py`. Logic lives in new `src/app/generator.py`
(`RAGGenerator`), with `LocalRetriever.agenerate` delegating. Generation
eval stays deferred. Citations are light-clamped to context `doc_path`s:
preserve model order, drop unknown paths with a log, keep the `grounded`
flag untouched. Transport and LLM errors are logged then raised, so only
model-unknown yields the fixed abstention (`grounded: false`); failures
never masquerade as abstentions. No truncation in v1: all context chunk
texts pass through in rank order.

## Alternatives considered

- Minimal pass-through prompt: simplest, lost on hallucination risk.
- Inline path markers in `str`: simplest return, lost on typed citations.
- Full file content for context: complete, lost on tokens and I/O since
  payload texts are already ranked.
- `GenerationPrompt` in `generator.py`: contained, lost on shared
  placement precedent (`Chunk`, `SearchHit` already live in schemas).
- Verbatim citations: preserves model intent exactly, lost on
  referential integrity because citation lists hallucinate paths;
  clamping keeps the relevance judgement while guaranteeing every cited
  path was provided.
- Abstention on error: typed flow with no caller branches, lost on
  silent failures because transport errors become indistinguishable
  from model-unknown; raising keeps the error signal explicit.
- Character-budget truncation: predictable size and cost, lost on
  information loss and a new tuning knob; pass-through keeps v1 minimal
  since typical hit sets sit far below the worst case.
- LLM-judge answer eval now: closes loop, lost on scope; deferred.

## Consequences

What becomes easier? Typed grounding without harness change; prompt
unit-testable via `build_prompt`; Slice 3 can call retrieve-then-generate
or one-shot `agenerate`; every cited path is guaranteed present in the
provided context. Harder? `SearchHit` in the protocol couples
harness types to chunk shape; unanswerable handling relies on model
self-report (`grounded: false`) until a judge lands; callers must
`try/except` around `agenerate` since transport failures raise instead
of returning abstentions.

## Rationale

Few-shot controls style without large token cost; chunk texts avoid disk
reads; structured citations give the future judge something to score;
schemas placement matches existing dataclass-in-schemas precedent;
deferring eval keeps Slice 2 small with Slice 3 unblocked.
