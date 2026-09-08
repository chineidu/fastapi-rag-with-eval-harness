---
status: ratified
date: 2026-09-08
deciders: user
---

# Naive fixed-size character chunker as RAG baseline

## Context

Phase 3 baseline needs a simple, deterministic chunker to measure retrieval
before structure-aware v2, hybrid search, and reranking. Corpus is FastAPI
markdown plus python examples on disk.

## Decision

Use fixed-size character slicing with sliding-window overlap. `DEFAULT_CHUNK_SIZE = 2000`
(500 tokens x 4 chars). API is `chunk_text(text, chunk_size, overlap=0) -> list[str]`
plus `chunk_directory(path, chunk_size, overlap=0) -> list[Chunk]`. Discovery is
recursive `rglob("*")` filtered to `{".md", ".py"}`, sorted, skipping empty.
`Chunk` stores `chunk_id`, `chunk_index`, `text`, `start_char`,
`end_char`, `token_count`, `doc_path` where `token_count` is `ceil(len/4)` min 1 for non-empty.
`doc_path` is ROOT-relative else input-relative. `overlap >= chunk_size` clamps
to `chunk_size // 2` with a warning log. Symlinked files are deduped by resolved
path (not by content); `doc_path` is the resolved target.

## Alternatives considered

- Tokenizer-based splitting: precise counts but adds a dependency for a baseline, so it lost.
- Markdown header-aware splitting: better boundaries but v2 scope per plan, so it lost.
- Single-file path API: simpler but mismatches directory eval flow, so it lost.
- Label-the-symlink (`link.md` kept as `doc_path`): preserves how the file was
  found, but splits identity for the same bytes across runs and complicates
  recall scoring, so resolved-target labeling won.

## Consequences

Easier: deterministic, no new deps, eval-traceable via stable IDs and offsets.
Harder: splits mid-sentence and mid-code, char proxy misestimates true tokens,
large overlap requests silently halve plus log.

## Rationale

Baseline must be minimal and comparable. Char slicing with a 4-char proxy keeps
the plan 500-token intent without a tokenizer, and stored offsets plus IDs let
the adapter map chunks back to docs for recall scoring under ADR-0012.
