"""Command-line interface for building the document index (``rag-index``)."""

from typing import Annotated

import typer

from src import ROOT
from src.app.indexer import Indexer
from src.app.vector_store import get_vector_store
from src.config import load_app_config
from src.embeddings import get_embedder

app = typer.Typer(
    name="rag-index",
    help="Build and inspect the document vector index.",
    no_args_is_help=True,
)

# Corpus roots from ADR-0002: English markdown docs plus Python examples.
DEFAULT_CORPUS_ROOTS: tuple[str, ...] = (
    "docs/fastapi/docs/en/docs",
    "docs/fastapi/docs_src",
)


@app.command()
def build(
    corpus: Annotated[
        list[str] | None,
        typer.Option(
            "--corpus",
            help="Corpus root directory, relative to repo root (repeatable)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild even if already indexed"),
    ] = False,
    config_path: Annotated[
        str,
        typer.Option("--config", help="App config YAML path"),
    ] = "",
) -> None:
    """Chunk the corpus, embed, and upsert into Qdrant."""
    # Load config (CLI path overrides the bundled default).
    cfg = load_app_config(config_path or None)
    embedder = get_embedder(cfg.embeddings_config)
    store = get_vector_store(cfg.indexer_config)
    indexer = Indexer(
        embedder,
        store,
        chunk_size=cfg.indexer_config.chunk_size,
        overlap=cfg.indexer_config.overlap,
    )
    # Reject empty path entries, which would resolve to ROOT itself.
    requested = corpus or list(DEFAULT_CORPUS_ROOTS)
    if any(not path.strip() for path in requested):
        typer.echo("error: corpus path must not be empty", err=True)
        raise typer.Exit(code=2)
    # Resolve and validate the corpus paths (default: both corpus roots).
    roots = [ROOT / path for path in requested]
    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        typer.echo(
            f"error: corpus path(s) do not exist: {', '.join(missing)}", err=True
        )
        raise typer.Exit(code=2)
    # Build and report.
    count = indexer.build(roots, force=force)
    typer.echo(
        f"Indexed {count} chunks into collection {cfg.indexer_config.qdrant.collection}"
    )


def main() -> None:
    """Console script entry point: delegate to the typer app."""
    app()


if __name__ == "__main__":
    main()
