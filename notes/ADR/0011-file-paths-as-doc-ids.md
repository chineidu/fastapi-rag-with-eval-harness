---
status: ratified
date: 2026-08-02
deciders: project owner
---

# File paths as document identifiers

## Context

The eval set, ground truth, and adapter all need to refer to
specific documents. The identifier scheme affects robustness to
content reorganization and ease of use across the pipeline.

## Decision

Use **file paths** as document identifiers (e.g.,
`docs/en/docs/tutorial/body.md`). The adapter internally derives
slugs if needed; the harness only operates on paths.

## Alternatives considered

- **Slugs** (e.g., `body` or `tutorial-body`): can collide
  across directories; change when content is reorganized;
  require a registry to resolve.
- **Hashes** (e.g., SHA256 of content): stable across moves
  but unreadable; break when content changes even slightly.

## Consequences

Easier: paths are stable, universal, and unambiguous in a
file-based corpus; humans can read them; no registry needed.

Harder: paths depend on directory layout; if the corpus is
reorganized (e.g., `docs/tutorial/` → `docs/learn/tutorial/`),
ground truth must be updated.

## Rationale

Paths are the canonical identifier in a file-based corpus.
Slugs can collide (`body.md` in two directories) or change when
content is reorganized. Hashes are stable but unreadable. Paths
are stable *enough* (the corpus is a snapshot, not a live tree)
and universal enough (no special resolver needed) to be the
right choice. The adapter can derive slugs on top of paths if
downstream code needs them.
