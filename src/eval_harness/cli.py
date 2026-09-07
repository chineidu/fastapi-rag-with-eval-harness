"""Command-line interface for the eval harness (``rag-eval``).

Exposes ``run``, ``diff``, and ``list`` subcommands backed by the harness
modules. The console script entry point ``rag-eval`` is declared in
``pyproject.toml`` under ``[project.scripts]``.

Exit codes
----------
``run``
    0 complete, 1 partial (at least one query failed), 2 usage/config error.
``diff``
    0 no significant regression, 1 at least one category regressed beyond
    threshold, 2 usage/run-not-found error.
``list``
    0 always (on success).
"""

import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, cast

import typer

from src.eval_harness.adapter import RetrieverAdapter
from src.eval_harness.config import apply_cli_overrides, load_harness_config
from src.eval_harness.loader import GroundTruthLoader
from src.eval_harness.metrics import OVERALL_CATEGORY
from src.eval_harness.runner import EvalRunner
from src.eval_harness.store import ResultStore, RunStatus
from src.schemas.containers import HarnessConfig
from src.schemas.harness import RunSummary
from src.schemas.types import DiffFlagEnum

app = typer.Typer(
    name="rag-eval",
    help="Evaluate retrieval quality before and after RAG pipeline changes.",
    no_args_is_help=True,
)

_CATEGORY_ORDER: tuple[str, ...] = ("DIRECT_LOOKUP", "MULTI_HOP", "CONCEPTUAL")


def load_adapter(import_path: str) -> RetrieverAdapter:
    """Import and instantiate an adapter from a ``module:Class`` path.

    Parameters
    ----------
    import_path : str
        Either ``"module.path:ClassName"`` or ``"module.path.ClassName"``.
        ``ClassName`` may itself be dotted (``"module:pkg.Class"``).

    Returns
    -------
    RetrieverAdapter
        A freshly instantiated adapter with a callable ``retrieve()``.

    Raises
    ------
    ValueError
        If ``import_path`` is empty or malformed.
    ImportError
        If the module or attribute cannot be found.
    TypeError
        If the target is not callable, cannot be instantiated with no
        arguments, or the instance lacks a callable ``retrieve()``.

    """
    if not import_path:
        raise ValueError("Adapter import path is empty")

    # Parse the import path.
    if ":" in import_path:
        module_name, _, attr_path = import_path.partition(":")
    else:
        module_name, _, attr_path = import_path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"Malformed adapter import path: {import_path!r}")

    # Import the module.
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import module {module_name!r} from {import_path!r}: {exc}"
        ) from exc

    # Traverse the module to find the target.
    try:
        target = module
        for part in attr_path.split("."):
            target = getattr(target, part)
    except AttributeError as exc:
        raise ImportError(
            f"Attribute {attr_path!r} not found in module {module_name!r}"
        ) from exc
    if not callable(target):
        raise TypeError(f"Adapter target {import_path!r} is not callable")

    # Instantiate the adapter.
    constructor: Callable[..., Any] = cast(Callable[..., Any], target)
    try:
        instance: object = constructor()
    except TypeError as exc:
        raise TypeError(
            f"Could not instantiate adapter {import_path!r} "
            f"(does it take a no-argument constructor?): {exc}"
        ) from exc

    # Verify duck-typing.
    if not callable(getattr(instance, "retrieve", None)):
        raise TypeError(f"Adapter {import_path!r} has no callable retrieve() method")
    return cast(RetrieverAdapter, instance)


def _config_with_overrides(
    *,
    config_path: str = "",
    adapter: str | None = None,
    ground_truth: str | None = None,
    db: str | None = None,
    k: int | None = None,
    concurrency: int | None = None,
    threshold_absolute: float | None = None,
    threshold_relative: float | None = None,
) -> HarnessConfig:
    """Load harness config and apply CLI overrides (CLI wins)."""
    base = load_harness_config(config_path or None)
    return apply_cli_overrides(
        base,
        adapter=adapter,
        ground_truth=ground_truth,
        db=db,
        k=k,
        concurrency=concurrency,
        threshold_absolute=threshold_absolute,
        threshold_relative=threshold_relative,
    )


def _fail(message: str) -> None:
    """Print an error to stderr and exit with code 2."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=2)


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Render an ASCII-aligned table with a dashed separator row."""
    widths = [len(column) for column in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format(cols: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(cols, widths)
        ).rstrip()

    lines = [_format(header), "  ".join("-" * width for width in widths)]
    lines.extend(_format(row) for row in rows)
    return "\n".join(lines)


def _print_run_summary(summary: RunSummary, k: int) -> None:
    """Print a run summary with per-category recall means."""
    typer.echo(f"Run {summary.run_id} ({summary.tag}): {summary.status}")
    typer.echo(
        f"  {summary.completed}/{summary.total} queries completed, "
        f"{summary.failed} failed, {summary.unanswerable} unanswerable"
    )
    if not summary.category_scores:
        return
    rows = []
    for category in _ordered_categories(set(summary.category_scores)):
        scores = summary.category_scores[category]
        rows.append([category, f"{scores['count']:.0f}", f"{scores['recall']:.3f}"])
    typer.echo(_table(["CATEGORY", "COUNT", f"RECALL@{k}"], rows))


def _category_means(
    results: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], int]:
    """Compute per-category recall means, excluding unanswerable rows.

    Unanswerable queries are stored with an empty ``ground_truth`` list and a
    recall of 0.0. Including them would drag the means to zero and show phantom
    regressions, so they are excluded here, mirroring ``EvalRunner``.

    Returns
    -------
    tuple[dict[str, dict[str, float]], int]
        ``(category -> {"recall", "count"}, excluded_count)``.
        No ``OVERALL`` entry is added when there are no answerable rows.

    """
    per_category: dict[str, list[float]] = {}  # {category: [recall@k, ...]}
    excluded = 0
    for row in results:
        if row["recall_at_k"] is None:
            continue
        try:
            ground_truth = json.loads(row["ground_truth"])
        except json.JSONDecodeError, TypeError:
            ground_truth = []
        if not ground_truth:
            excluded += 1
            continue
        per_category.setdefault(row["category"], []).append(float(row["recall_at_k"]))

    means: dict[str, dict[str, float]] = {}
    all_values: list[float] = []
    for category, values in per_category.items():
        means[category] = {
            "recall": sum(values) / len(values),
            "count": float(len(values)),
        }
        all_values.extend(values)
    if all_values:
        means[OVERALL_CATEGORY] = {
            "recall": sum(all_values) / len(all_values),
            "count": float(len(all_values)),
        }
    return means, excluded


def _answerable_recalls(results: list[dict[str, Any]]) -> dict[str, float]:
    """Map query_id to recall@k for answerable, successful rows only."""
    out: dict[str, float] = {}
    for row in results:
        if row["recall_at_k"] is None:
            continue
        try:
            ground_truth = json.loads(row["ground_truth"])
        except json.JSONDecodeError, TypeError:
            ground_truth = []
        if not ground_truth:
            continue
        out[row["query_id"]] = float(row["recall_at_k"])
    return out


def _ordered_categories(categories: set[str]) -> list[str]:
    """Order known categories first, other categories alphabetically, OVERALL last."""
    known: list[str] = [
        category for category in _CATEGORY_ORDER if category in categories
    ]
    other: list[str] = sorted(
        category
        for category in categories
        if category not in _CATEGORY_ORDER and category != OVERALL_CATEGORY
    )
    if OVERALL_CATEGORY in categories:
        return known + other + [OVERALL_CATEGORY]
    return known + other


def _format_pct(delta: float, baseline: float) -> str:
    """Format a percent delta, or ``n/a`` when the baseline is zero."""
    if baseline > 0:
        # `+.1f`: show the sign explicitly (+/-) and round to one decimal place.
        return f"{((delta / baseline) * 100):+.1f}%"
    return "n/a"


def _significance(
    delta: float,
    baseline: float,
    threshold_absolute: float,
    threshold_relative: float,
) -> DiffFlagEnum | None:
    """Classify a delta as a ``DiffFlagEnum`` member, or ``None`` when noise.

    A delta is significant when its magnitude meets either the absolute or the
    relative threshold; relative comparison is skipped when ``baseline`` is 0.
    """
    if abs(delta) < 1e-9:
        return None
    significant = abs(delta) >= threshold_absolute or (
        baseline > 0 and ((abs(delta) / baseline) * 100) >= threshold_relative
    )
    if not significant:
        return None
    return DiffFlagEnum.REGRESSED if delta < 0 else DiffFlagEnum.IMPROVED


def _render_verbose_diff(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> None:
    """Print per-query recall deltas for answerable queries."""
    baseline = _answerable_recalls(baseline_rows)
    current = _answerable_recalls(current_rows)
    rows = []
    for query_id in sorted(set(baseline) | set(current)):
        base = baseline.get(query_id, 0.0)
        curr = current.get(query_id, 0.0)
        rows.append([query_id, f"{base:.3f}", f"{curr:.3f}", f"{curr - base:+.3f}"])
    typer.echo("\nPer-query deltas (answerable queries only):")
    typer.echo(_table(["QUERY_ID", "BASELINE", "CURRENT", "DELTA"], rows))


def _render_diff(
    baseline_run: dict[str, Any],
    current_run: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    cfg: HarnessConfig,
    verbose: bool,
) -> bool:
    """Print the before/after comparison table; return True if any regression."""
    typer.echo(
        f"diff {baseline_run['tag']} (#{baseline_run['id']}) -> "
        f"{current_run['tag']} (#{current_run['id']})"
    )
    for run in (baseline_run, current_run):
        if run["status"] != RunStatus.COMPLETE.value:
            typer.echo(
                f"warning: run #{run['id']} ({run['tag']}) is {run['status']}, "
                "not complete",
                err=True,
            )
    baseline_means, baseline_excluded = _category_means(baseline_rows)
    current_means, current_excluded = _category_means(current_rows)
    if baseline_excluded or current_excluded:
        typer.echo(
            f"note: excluded {baseline_excluded}/{current_excluded} unanswerable "
            "rows (baseline/current) from means",
            err=True,
        )
    categories = _ordered_categories(set(baseline_means) | set(current_means))
    if not categories:
        typer.echo("No answerable queries found in either run to compare.")
        return False
    rows = []
    regressed = False
    for category in categories:
        base = baseline_means.get(category, {"recall": 0.0, "count": 0})
        curr = current_means.get(category, {"recall": 0.0, "count": 0})
        delta = curr["recall"] - base["recall"]
        flag = _significance(
            delta,
            base["recall"],
            cfg.diff.threshold_absolute,
            cfg.diff.threshold_relative,
        )
        if flag == DiffFlagEnum.REGRESSED:
            regressed = True
        rows.append(
            [
                category,
                f"{base['recall']:.3f}",
                f"{curr['recall']:.3f}",
                f"{delta:+.3f}",
                _format_pct(delta, base["recall"]),
                flag.value if flag is not None else "",
            ]
        )
    typer.echo(
        _table(["CATEGORY", "BASELINE", "CURRENT", "DELTA", "DELTA%", "FLAG"], rows)
    )
    if verbose:
        _render_verbose_diff(baseline_rows, current_rows)
    return regressed


@app.command()
def run(
    adapter: Annotated[
        str,
        typer.Option(
            "--adapter", help="Adapter import path, e.g. app.adapter:Retriever"
        ),
    ] = "",
    ground_truth: Annotated[
        str,
        typer.Option("--ground-truth", help="Ground truth JSON/JSONL path"),
    ] = "",
    tag: Annotated[
        str,
        typer.Option("--tag", help="Run label (auto-generated when omitted)"),
    ] = "",
    k: Annotated[
        int | None,
        typer.Option("--k", help="Top-k for recall@k"),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Max parallel adapter calls"),
    ] = None,
    db: Annotated[
        str,
        typer.Option("--db", help="SQLite database path"),
    ] = "",
    config_path: Annotated[
        str,
        typer.Option("--config", help="Harness config YAML path"),
    ] = "",
) -> None:
    """Evaluate the configured adapter over ground truth queries."""
    try:
        cfg = _config_with_overrides(
            config_path=config_path,
            adapter=adapter or None,
            ground_truth=ground_truth or None,
            db=db or None,
            k=k,
            concurrency=concurrency,
        )
    except ValueError as exc:
        _fail(str(exc))
    if not cfg.adapter:
        _fail("no adapter configured: pass --adapter or set it in .rag-eval.yaml")
    try:
        retriever = load_adapter(cfg.adapter)
    except (ImportError, TypeError, ValueError) as exc:
        _fail(str(exc))
    try:
        records = GroundTruthLoader.load(cfg.ground_truth)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    if not records:
        _fail(f"no ground truth records in {cfg.ground_truth}")
    if not tag:
        tag = f"auto-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}"
    with ResultStore(cfg.db) as store:
        eval_runner = EvalRunner(
            retriever,
            store,
            records,
            k=cfg.defaults.k,
            concurrency=cfg.defaults.concurrency,
        )
        summary = eval_runner.run(
            tag,
            config={
                "adapter": cfg.adapter,
                "k": cfg.defaults.k,
                "concurrency": cfg.defaults.concurrency,
            },
        )
        _print_run_summary(summary, cfg.defaults.k)
    if summary.failed > 0:
        raise typer.Exit(code=1)


@app.command()
def diff(
    baseline: Annotated[
        str,
        typer.Argument(help="Baseline run reference (TAG or numeric id)"),
    ],
    current: Annotated[
        str,
        typer.Argument(help="Current run reference (TAG or numeric id)"),
    ],
    threshold_absolute: Annotated[
        float | None,
        typer.Option("--threshold-absolute", help="Flag when |delta| exceeds this"),
    ] = None,
    threshold_relative: Annotated[
        float | None,
        typer.Option(
            "--threshold-relative",
            help="Flag when |delta%| exceeds this (percent)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show per-query deltas"),
    ] = False,
    db: Annotated[
        str,
        typer.Option("--db", help="SQLite database path"),
    ] = "",
    config_path: Annotated[
        str,
        typer.Option("--config", help="Harness config YAML path"),
    ] = "",
) -> None:
    """Compare two runs and flag significant category regressions."""
    try:
        cfg = _config_with_overrides(
            config_path=config_path,
            db=db or None,
            threshold_absolute=threshold_absolute,
            threshold_relative=threshold_relative,
        )
    except ValueError as exc:
        _fail(str(exc))
    with ResultStore(cfg.db) as store:
        try:
            baseline_run = store.resolve_ref(baseline)
            current_run = store.resolve_ref(current)
        except LookupError as exc:
            _fail(str(exc))
        regressed = _render_diff(
            baseline_run,
            current_run,
            store.get_results(baseline_run["id"]),
            store.get_results(current_run["id"]),
            cfg,
            verbose,
        )
    if regressed:
        raise typer.Exit(code=1)


@app.command(name="list")
def list_runs(
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum runs to show"),
    ] = 50,
    db: Annotated[
        str,
        typer.Option("--db", help="SQLite database path"),
    ] = "",
    config_path: Annotated[
        str,
        typer.Option("--config", help="Harness config YAML path"),
    ] = "",
) -> None:
    """List recent runs stored in the SQLite database."""
    cfg = _config_with_overrides(config_path=config_path, db=db or None)
    with ResultStore(cfg.db) as store:
        runs = store.list_runs(limit)
    if not runs:
        typer.echo("No runs yet.")
        return
    rows = [
        [
            str(row["id"]),
            row["tag"],
            row["created_at"],
            row["status"],
            f"{row['succeeded']}/{row['total']}",
        ]
        for row in runs
    ]
    typer.echo(_table(["ID", "TAG", "CREATED", "STATUS", "QUERIES"], rows))


def main() -> None:
    """Console script entry point: delegate to the typer app."""
    app()


if __name__ == "__main__":
    main()
