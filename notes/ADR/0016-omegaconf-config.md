---
status: ratified
date: 2026-08-02
deciders: project owner
---

# OmegaConf for configuration

## Context

The harness needs a config system for defaults (k, concurrency,
db path, thresholds) that supports both a committed config file
(`.rag-eval.yaml`) and CLI overrides.

## Decision

Use **OmegaConf** (structured configs, dataclass-based) with
file → CLI merge order:

1. Parse `.rag-eval.yaml` (default path: project root,
   committed).
2. Apply CLI overrides (`--k 20 --concurrency 5`).
3. CLI wins.

Config schema:

```yaml
# .rag-eval.yaml
adapter: myproject.adapter:LocalRetriever
ground_truth: data/ground_truth.jsonl
db: data/.rag-eval/runs.db

defaults:
  k: 10
  concurrency: 3

diff:
  threshold_absolute: 0.05
  threshold_relative: 5
```

## Alternatives considered

- **Plain YAML + manual parsing**: no type safety; no merge
  semantics; re-invents what OmegaConf does.
- **Pydantic settings**: similar type safety but heavier;
  OmegaConf has cleaner CLI merge via `from_cli()`.
- **argparse only**: forces everything into flags; no committed
  defaults.

## Consequences

Easier: structured configs give runtime type safety (`k: int`
can't accidentally become a string); `OmegaConf.from_cli()`
auto-parses `--k 20 --concurrency 5` without manual argparse
boilerplate; merge semantics match the harness's needs.

Harder: OmegaConf is a dependency (already used in this
engineer's other projects, so the cost is paid).

## Rationale

OmegaConf is already used in this engineer's other projects —
consistency matters. Structured configs give runtime type safety
that plain YAML can't. Merge semantics (file → CLI) map directly
to the harness's needs. The CLI parser is built in.
