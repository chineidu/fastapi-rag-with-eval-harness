---
status: proposed
date: 2026-09-16
deciders: project owner
amends: 0004-retriever-adapter-protocol
---

# Streaming generation over SSE

## Context

Slice 3b must deliver long answers without stalling time-to-first-byte,
over the same service wiring as 3a. The generation contract guarantees
citation-clamped structured answers, and streaming must preserve that
guarantee rather than downgrade to free text. Two delivery shapes were
on the table.

## Decision

Stream `instructor.Partial[GeneratedAnswer]` snapshots as server-sent
events from a separate `POST /ask/stream` endpoint. `RAGGenerator`
gains `astream`, which yields growing partials and appends the
citation-clamped final answer; `LocalRetriever` gains `astream`, which
retrieves once up front and forwards the generator stream. Mid-stream
snapshots may carry incomplete citations; only the last data event
before `[DONE]` is clamped. Pre-first-byte failures keep their 3a
status codes (422, 504, 500); a mid-stream stall terminates the
connection without an envelope. `api_config.timeout` bounds
time-to-first-token.

## Alternatives considered

- Raw OpenAI text stream with a terminal metadata event: simpler bytes
  with no partial-schema machinery, but citations stay unknowable until
  generation ends and it promotes `openai` to a direct dependency; lost
  to contract preservation.
- Multiplexed `/ask` with a stream flag: one route instead of two, but
  content-type negotiation complicates caching and clients, and
  ADR-0028 already reserves a separate endpoint; lost to the ratified
  layout.

## Consequences

What becomes easier? Long answers start arriving immediately while the
full structured answer still lands at the end; no new dependency, since
`instructor` already provides partial streaming.

What becomes harder? Clients must accumulate snapshots and treat only
the pre-`[DONE]` event as final; partial citations must never render as
authoritative. Mid-stream failures have no envelope by construction.

What new obligations? Any future streaming endpoint reuses the
partials-then-clamped-final shape; the clamp helper stays shared
between `agenerate` and `astream` so the two paths cannot disagree on
citation validity.

## Rationale

Partial snapshots preserve the grounded-answer contract end to end
with zero new dependencies, and the separate endpoint keeps 3a clients
untouched. Timeout scoping to first-token matches the failure clients
can actually act on: before bytes flow the status code still carries
meaning, after that only the connection state does.
