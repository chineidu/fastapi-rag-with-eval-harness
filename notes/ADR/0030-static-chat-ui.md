---
status: ratified
date: 2026-09-17
deciders: project owner
---

# Bundled static chat UI

## Context

Slice 3 delivered the service with a streaming endpoint, but the only
clients are curl and the test suite, so the streaming experience is
invisible and every manual check needs hand-built HTTP requests. The
repo has no frontend toolchain, no build step, and no Node
dependencies, and Slice 4 assumes a single deployable image, so a
browser client must justify any new stack it adds.

## Decision

Serve one self-contained `index.html` from the API at `/`, with the
route excluded from the OpenAPI schema. The page is vanilla HTML, CSS,
and JS with no CDN references and no build step; it talks to
`POST /api/v1/ask/stream` via `fetch` plus a stream reader, since
`EventSource` cannot send a JSON POST body. Frames render as they
arrive by rebuilding the answer from each cumulative snapshot, with
light client-side formatting (paragraphs, lists, inline code, and
bracketed `[doc.md]` citations as chips) constructed from DOM nodes;
the last frame before `[DONE]` supplies citations and model id.
Same-origin serving means no CORS involvement. The asset lives in
`src/api/static/`, served by an unversioned `src/api/routes/ui.py`
router.

## Alternatives considered

- React SPA under `ui/`: richer component model and portfolio
  familiarity, but adds a Node toolchain, a build pipeline, and a
  second deployable; lost to the single-image scope.
- Streamlit or Gradio app: fastest Python-only interactivity, but a
  separate runtime, a new dependency, and it would not ship with the
  service; lost to keeping one artifact.
- No UI for now: keeps focus on Slice 4, but leaves the streaming
  endpoint unexperienced and the pre-first-byte UX decision abstract;
  lost to the demo and interactivity need.

## Consequences

Easier: the service is demoable in a browser, streaming and citations
become concrete, and manual verification stops needing curl.

Harder: the page adds an unversioned surface that must not collide with
the API prefix, and its JS cannot be unit-tested by pytest, so the
client logic (including the snapshot formatter) must stay minimal and
the route behavior stays the tested part.

New obligations: keep the page dependency-free and build model output
through DOM text nodes only, never `innerHTML`; if the API prefix
becomes configurable at runtime, the page's hardcoded `/api/v1` paths
must be updated with it.

## Rationale

The page is the smallest change that makes the delivered surface
visible: no new dependency, no second artifact, same-origin, and it
reuses the exact SSE contract the streaming tests already pin. The
heavier frontends buy polish this project does not need yet, and the
SSE mechanics are identical in any browser client if it does later.
