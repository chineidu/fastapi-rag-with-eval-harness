---
status: ratified
date: 2026-09-15
deciders: architect (user) + builder (agent)
---

# API service structure

## Context

Slice 3 needs an HTTP service over retrieve plus generate, in two parts:
a non-streaming endpoint first, then a separate streaming endpoint on
the same service wiring. The service follows an established internal
API layout: an app factory with prefix normalization, `core/` modules for exceptions,
lifespan, middleware, dependencies, rate limiting, and responses, plus
versioned route packages. This repo already carries compatible pieces:
`api_config` in YAML, `ErrorCodeEnum` in shared types, `msgspec` as a
direct dependency, and per-environment `LIMIT_VALUE` settings. Three
points the pattern does not settle for this repo were decided
2026-09-15: rate limiting, route versioning, and readiness scope.

## Decision

Adopt this structure for `src/api/`: `app.py` with
`create_application`, prefix normalization, CORS, and a middleware
stack; `core/` with exceptions, lifespan, middleware, dependencies,
and response helpers; `routes/v1/` with versioned routers mounted
under the normalized base prefix. Port it slimmed down: no threadpool
executor (the service is fully async), and lifespan builds
`LocalRetriever` plus `RAGGenerator` clients into `app.state` instead
of model loaders. Ship liveness and readiness (Qdrant reachable, index
present) in 3a. Defer slowapi rate limiting to deploy time. Extend
`ErrorCodeEnum` with generation, timeout, and input codes while keeping
the `{status, error, request_id, path}` envelope and the
`MsgSpecJSONResponse` renderer.

## Alternatives considered

- Flat unversioned router: less scaffolding now, but versioning
  retrofit later touches every route import; lost to v1-from-day-one.
- Adopt slowapi in 3a: full pattern parity, but a new dependency and
  lockfile churn for local/dev use with no deploy need; lost to
  deferral alongside Slice 4 distribution work.
- Health-only in 3a: smaller surface, but Slice 4 already plans
  readiness-gated replica boots, so the probe point is free now;
  lost to shipping readiness early.

## Consequences

Versioning makes v2 additive later instead of a rewrite. Deferring
rate limiting keeps 3a dependency-free while composing with the
deploy-time distribution ADR. Readiness probes written now get reused
by Slice 4 replica boot checks. New obligations: the port must drop
threadpool and executor idioms rather than copying them blindly, and
`APIConfig` needs a request `timeout` field (a 3a contract detail).

## Rationale

Pattern parity where it is free (msgspec, config, and settings pieces
already exist here), deferral where it costs without benefit
(slowapi before any deploy), and readiness now because the
distribution design already assumes it. The versioned layout matches
the committed `/api/v1` prefix instead of fighting it.
