# Typer CLI guide for scripts

How to structure Typer apps in `scripts/`, and when to use `@app.callback()`.

## App vs command

A Typer app has two layers:

1. **The app** (`typer.Typer(...)`) — the top-level CLI entrypoint
2. **Commands** (`@app.command()`) — the subcommands the user runs

```text
uv run -m scripts.classify_eval_data classify --input-path ...
         └────────── app ──────────┘ └── command ──┘
```

```python
import typer

app = typer.Typer(help="Short description of the script", add_completion=False)


@app.command()
def classify(...) -> None:
    """Do the real work."""
    ...


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
```

## What `@app.callback()` is

`@app.callback()` is a hook that runs **when the app is invoked**, before any command. Use it for shared setup that should happen for every command (logging init, config checks, auth).

### Bad example: empty callback

An empty callback adds nothing and should be omitted.

```python
@app.callback()
def _main_callback() -> None:
    """Classify eval records into labels."""
    # empty body — no shared setup, no purpose
```

### Good example: logging init

Use the callback to configure logging once before any command runs.

```python
import logging

from src import create_logger

logger = create_logger(name=__name__)


@app.callback()
def _main_callback(verbose: bool = False) -> None:
    """Set up logging and run before any command."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)
    logger.info("starting fetch script")
```

Then it works for both commands:

```bash
uv run -m scripts.fetch_eval_data github --verbose
uv run -m scripts.fetch_eval_data stackoverflow --verbose
```

### Good example: shared env validation

A multi-command API fetch script can validate tokens once in the callback.

```python
from src.config import app_settings


@app.callback()
def _main_callback() -> None:
    """Verify required credentials before running any command."""
    if not app_settings.GITHUB_READ_ACCESS:
        raise typer.BadParameter("GITHUB_READ_ACCESS is required")
    if not app_settings.STACK_EXCHANGE_READ_ACCESS:
        logger.warning("STACK_EXCHANGE_READ_ACCESS not set; lower rate limits apply")
```

## When a callback is useful

**Multi-command app** — shared setup across commands:

```bash
uv run -m scripts.fetch_eval_data github ...
uv run -m scripts.fetch_eval_data stackoverflow ...
```

A callback would run for both `github` and `stackoverflow`. Useful if both need the same preamble.

**Single-command app** — only if it does real work:

```bash
uv run -m scripts.classify_eval_data classify ...
```

There is only one command, so the callback is less compelling. Add one only if it performs a shared setup step (logging, validation, etc.). An empty callback adds nothing.

## Project conventions

| Script | Commands | Uses `@app.callback()`? |
|---|---|---|
| `fetch_eval_data.py` | `github`, `stackoverflow` | Yes — logs the invoked command |
| `normalize_eval_data.py` | `normalize` | Yes — logs the invoked command |
| `classify_eval_data.py` | `classify` | Yes — logs the invoked command |

Rules of thumb:

1. Give every callback a **real operation** (logging, env validation, shared config). Empty callbacks should be removed.
2. Put the real work in `@app.command()` functions.
3. Keep `add_completion=False` unless shell completion is intentionally supported.
4. Expose the app via `_main()` → `app()` under `if __name__ == "__main__":`.

## Invocation style

Always document and prefer module form:

```bash
uv run -m scripts.<module_name> <command> [OPTIONS]
```

Examples:

```bash
uv run -m scripts.fetch_eval_data github --num 30
uv run -m scripts.normalize_eval_data normalize
uv run -m scripts.classify_eval_data classify
```
