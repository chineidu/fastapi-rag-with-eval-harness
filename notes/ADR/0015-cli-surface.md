---
status: ratified
date: 2026-08-02
deciders: project owner
---

# CLI surface: run, diff, list

## Context

The harness needs a CLI for invocation. The command surface
should cover the three main workflows: run an eval, diff two
runs, and list past runs.

## Decision

Three subcommands, exposed as `rag-eval`:

```text
rag-eval run [OPTIONS]
rag-eval diff TAG|ID TAG|ID [OPTIONS]
rag-eval list
```

#### `rag-eval run`

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--adapter` | yes | - | Import path to adapter class |
| `--ground-truth` | yes | - | Path to enriched JSONL file |
| `--tag` | no | auto | Label for this run |
| `--k` | no | 10 | Top-k for recall@k |
| `--concurrency` | no | 3 | Max parallel adapter calls |
| `--db` | no | `data/.rag-eval/runs.db` | SQLite path |

#### `rag-eval diff`

| Flag | Default | Notes |
|---|---|---|
| `--threshold-absolute` | 0.05 | Flag if \|Δ\| exceeds |
| `--threshold-relative` | 5 | Flag if \|Δ%\| exceeds (percent) |

Exit codes:

- `0` — no significant regressions.
- `1` — at least one category regressed beyond threshold.
- `2` — diff failed (missing run, bad args).

## Alternatives considered

- **Single command with subcommand-as-flag**: harder to discover;
  every flag appears in `--help`.
- **No CLI, Python API only**: blocks non-Python usage; less
  scriptable for CI.

## Consequences

Easier: each workflow has a clear verb; flags are scoped to the
relevant subcommand; exit codes enable CI integration.

Harder: three subcommands means three sets of flags; help text
needs to be clear.

## Rationale

Three verbs map cleanly to the three workflows (run, compare,
browse). Subcommands keep flag scope narrow. Exit codes are
critical for CI: a regression should be a non-zero exit so CI
fails the PR.
