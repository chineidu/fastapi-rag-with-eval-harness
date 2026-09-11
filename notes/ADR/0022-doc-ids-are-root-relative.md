---
status: ratified
date: 2026-09-11
deciders: project owner
---

# Document IDs are ROOT-relative paths

## Context

ADR-0011 chose file paths as document identifiers but did not define the
base they are relative to. The labeling pipeline and the indexer derived
paths independently and disagreed. `scripts/label_ground_truth.py` used
`relative_to(root.parent.parent.parent)`, which is depth-dependent and
resolved to the FastAPI checkout root for markdown (`docs/en/docs/...`) but
to the repo root for python (`docs/fastapi/docs_src/...`). The chunker
emits repo-root-relative paths for both (`docs/fastapi/docs/en/docs/...`).
The harness scores by exact string intersection, so 45 markdown refs in the
ground truth could never match an indexed document.

## Decision

Document IDs are POSIX paths relative to the repository root (`ROOT` in
`src/__init__.py`) for every corpus root:

- `docs/fastapi/docs/en/docs/tutorial/body.md` (markdown)
- `docs/fastapi/docs_src/dependencies/tutorial001.py` (python)

The chunker already emits this form (ADR-0018) and is unchanged.
`data/ground_truth.jsonl` markdown refs were rewritten from
`docs/en/docs/...` to `docs/fastapi/docs/en/docs/...` (identifier-only
rewrite; the LLM verdicts are unchanged, no re-labeling). `_load_corpus` in
`scripts/label_ground_truth.py` now resolves both roots with
`relative_to(ROOT)` and falls back to the input root when a corpus lives
outside the repository.

## Alternatives considered

- **Checkout-relative IDs** (relative to `docs/fastapi/`): matches
  ADR-0011's example and is independent of where the checkout is vendored,
  but requires rewriting the 83 python refs, adding a corpus-base parameter
  to the chunker, and changing ADR-0018 for more churn.
- **Adapter-side translation**: hides the data bug, makes the RAG adapter
  ground-truth-aware, and makes audit metadata report IDs that are absent
  from the index.
- **Re-label the ground truth**: costs LLM calls for a change that only
  rewrites identifier strings.

## Consequences

Easier: index, adapter, and ground truth share one ID space; no
normalization layer; markdown docs become scoreable.

Harder: IDs carry the in-repo checkout location (`docs/fastapi/`), so
moving the corpus within the repo would invalidate them. The labeling
embedding cache is invalidated by the path change and will re-embed on the
next labeling run.

## Rationale

The scoring layer compares raw strings, so the ID base is a data contract,
not an implementation detail. Repo-root-relative paths are what the chunker
already produces and what the harness stores; aligning the ground truth and
the labeling script to them is the smallest change that removes an entire
class of silent zero scores.
