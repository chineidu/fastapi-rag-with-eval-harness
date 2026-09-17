"""Server entrypoint: configure logging first, then serve the API."""

import logging
import os
from typing import Annotated

import typer

__all__ = ["main"]


app = typer.Typer(
    name="rag-serve",
    help="Serve the RAG HTTP API.",
    no_args_is_help=False,
    add_completion=False,
)


def _log_params_from_env() -> tuple[int, bool, str | None]:
    """Read LOG_LEVEL, LOG_STRUCTURED, and LOG_FILE from the environment.

    Returns
    -------
    tuple[int, bool, str | None]
        Parsed level, structured flag, and optional log file path. An
        unrecognized LOG_LEVEL falls back to INFO.

    """
    raw_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelNamesMapping().get(raw_level, logging.INFO)
    raw_structured = os.getenv("LOG_STRUCTURED", "").strip().lower()
    structured = raw_structured in {"1", "true", "yes"}
    log_file = os.getenv("LOG_FILE", "").strip() or None
    return level, structured, log_file


@app.command()
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind host (overrides settings)"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Bind port (overrides settings)"),
    ] = None,
) -> None:
    """Serve the API after configuring logging from the environment.

    The CLI flags override the settings values and are authoritative
    because the dotenv loader gives ``.env`` precedence over the shell
    environment. App modules create loggers at import time, so the
    configured ``create_logger`` call must run before they are
    imported; that is why the project imports live inside this
    function. Serve with ``uv run python -m src.main``.
    """
    level, structured, log_file = _log_params_from_env()
    from src import create_logger

    logger = create_logger(
        __name__, level=level, structured=structured, log_file=log_file
    )

    import uvicorn

    from src.api.app import create_app
    from src.app.adapter import LocalRetriever
    from src.config.settings import app_settings

    bind_host = host if host is not None else app_settings.HOST
    bind_port = port if port is not None else app_settings.PORT
    api = create_app(LocalRetriever())
    logger.info("Serving API on %s:%d", bind_host, bind_port)
    uvicorn.run(api, host=bind_host, port=bind_port)


def main() -> None:
    """Console script entry point: delegate to the typer app."""
    app()


if __name__ == "__main__":
    main()
