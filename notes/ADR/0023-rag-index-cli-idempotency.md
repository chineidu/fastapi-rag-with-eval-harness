---
status: ratified
date: 2026-09-12
deciders: project owner
---

# rag-index CLI surface and corpus-fingerprint build idempotency

## Context

The app side has a `rag-index` CLI with a single `build` command, used to
populate Qdrant before eval runs. It was added during Phase D without an
ADR; ADR-0015 covers only the harness's `rag-eval` surface. Two gaps
surfaced during review: the help text promised inspection that does not
exist, and the documented idempotency was misleading. `ensure_collection`
compares only `model_id`/`dim` in the sentinel meta point, so an unchanged
corpus is still re-chunked and re-embedded, while deleted or shrunk
documents leave stale points behind because chunk ids are
`doc_path#index` and content-independent.

## Decision

`rag-index` exposes two subcommands:

- `rag-index build [--corpus PATH ...] [--force] [--config PATH]`. Keeps
  the ADR-0002 default corpus roots, repeatable `--corpus` override, and
  exit code 2 for empty or missing corpus paths. `--force` drops and
  recreates the collection before indexing.
- `rag-index inspect [--config PATH]`. Read-only; prints backend,
  collection, recorded model id and dimension, and chunk count, with an
  explicit "not indexed" state.

Index idempotency becomes corpus-content aware. The sentinel meta point
records a corpus fingerprint: a SHA-256 over the sorted chunk ids and the
per-chunk text hashes, which covers file set, file contents, and chunking
parameters. `ensure_collection` reuses the collection only when
`model_id`, `dim`, and fingerprint all match and `--force` is absent;
otherwise it drops and recreates the collection, which also clears stale
points from removed or shrunk documents. On reuse, `build` skips
embedding and upsert entirely and reports the index as up to date rather
than claiming it indexed chunks.

The vector store contract is extended accordingly:
`ensure_collection(model_id, dim, *, fingerprint, force=False) -> bool`
(True when created or rebuilt, False when reused unchanged) and
`describe() -> CollectionInfo`.

## Alternatives considered

- Wording-only fix: correct the help text and docstrings, keep
  re-embedding and stale points. Rejected because stale points can
  silently distort eval numbers.
- Always rebuild on every run: correct, but no-op runs still pay the full
  embedding cost.
- Fingerprint from a pre-chunk file walk: same result, but duplicates the
  chunker's discovery logic (rglob, symlink dedupe, encoding skips).
- Un-typed `get_meta()` dict on the stack: rejected in favor of a typed
  `CollectionInfo` and explicit `IndexReport`.
- Amend ADR-0015: rejected. ADR-0015 is scoped to the harness CLI, and
  this decision also covers storage semantics, not just CLI surface.

## Consequences

Easier: no-op rebuilds are cheap and honest; `inspect` answers what model,
dimension, and count the index holds without ad-hoc scripts; stale points
stop accumulating. The first build after this change rebuilds any existing
collection once, because its meta point has no fingerprint.

Harder: the sentinel payload grows by one field; every `ensure_collection`
caller must supply a fingerprint; the store contract grows `describe()`
and a `_collection_name` hook. `build` now returns `IndexReport` instead
of an int, so callers and tests update.

## Rationale

The eval harness is only as trustworthy as the index it scores against.
Model/dim matching prevents vector-space mismatches but not stale corpus
content, so the fingerprint closes the gap with data the indexer already
materializes: the chunk list. Deriving it from chunks avoids a second file
walk and keeps one discovery path. A typed `IndexReport` lets the CLI tell
the truth about skip versus rebuild, which is the user-visible half of the
same gap. `inspect` makes the claimed read path real and gives D5+ work a
way to confirm model and count at a glance.
