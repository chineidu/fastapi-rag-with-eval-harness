---
status: ratified
date: 2026-08-02
deciders: project owner
---

# JSONL ground truth format

## Context

The ground truth file is read on every eval run. The labeling
pipeline that produces it is long-running (70 queries × 30
candidates × LLM judge) and crash-prone (network calls, rate
limits). The format must support crash-safe writes and resume,
and per-record validation.

## Decision

Use **JSONL** (one compact JSON object per line). The labeling
script writes one record per line as it completes each query,
enabling crash-safe incremental writes and resume-on-crash.

Schema fields:

| Field | Required | Purpose |
|---|---|---|
| `query_id` | yes | Unique ID matching the original source |
| `label` | yes | Difficulty category |
| `query_text` | yes | What the harness passes to `adapter.retrieve()` |
| `relevant_docs` | yes | File paths of docs answering the query (empty when `answerable: false`) |
| `answerable` | yes | Whether any corpus doc answers the query |
| `source` | no | `github` or `stackoverflow` |
| `title` | no | Human-readable title |
| `answer_text` | no | Reference answer (labeling context, not used in eval) |
| `url` | no | Source link |

## Alternatives considered

- **JSON array** (single document, all records): atomic writes
  only; any crash loses all progress; can't validate per-line.
- **SQLite for ground truth**: queryable but overkill; ground
  truth is read-only at eval time and JSONL is simpler to inspect
  and diff.

## Consequences

Easier: incremental writes; resume-on-crash; per-line validation;
human-readable (one record per line, easy to inspect with `jq` or
`head`).

Harder: line-oriented parsing needed; partial files on crash need
cleanup logic.

## Rationale

JSONL is the standard format for append-only data with crash-safe
writes — the labeler writes one line at a time and can resume
from the last completed line. Per-line validation lets the loader
skip malformed records without rejecting the whole file. The
file-path identifier in `relevant_docs` (see ADR-0011) is stable
across chunking changes.
