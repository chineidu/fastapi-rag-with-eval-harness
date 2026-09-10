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


@app.command()
def build(
    corpus: Annotated[
        str,
        typer.Option("--corpus", help="Corpus root directory (relative to repo root)"),
    ] = "docs/fastapi/docs/en/docs",
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
    # Resolve and validate the corpus path.
    root = ROOT / corpus
    if not root.exists():
        typer.echo(f"error: corpus path does not exist: {root}", err=True)
        raise typer.Exit(code=2)
    # Build and report.
    count = indexer.build(root, force=force)
    typer.echo(
        f"Indexed {count} chunks into collection {cfg.indexer_config.qdrant.collection}"
    )


def main() -> None:
    """Console script entry point: delegate to the typer app."""
    app()


if __name__ == "__main__":
    main()
