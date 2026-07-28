"""
Asynchronous, non-blocking logging configuration for the application.

This module provides a production-grade logging setup based on Python's
QueueHandler / QueueListener pattern. The primary goals are:

- Ensure that application threads (FastAPI requests, asyncio tasks, workers)
  never block on log I/O (stdout, files, JSON serialization).
- Centralize all log formatting and output in a single background thread.
- Enforce a single, process-wide logging configuration to avoid duplicated
  handlers, multiple listener threads, and inconsistent formats.

Architecture overview
---------------------

    Application code (async / sync)
                |
                v
        logging.Logger
                |
                v
        QueueHandler (non-blocking)
                |
                v
        Queue[LogRecord]
                |
                v
        QueueListener (background thread)
                |
                v
    StreamHandler / FileHandler (I/O + formatting)

Key design decisions
--------------------

- A single global Queue and QueueListener are created per process.
- The first call to `create_logger()` initializes the logging system
  ("first caller wins").
- Structured (JSON) vs plain-text logging is a process-wide decision and
  cannot be changed after initialization.
- All real handlers (console, file, etc.) are attached to the QueueListener,
  never directly to loggers, to guarantee non-blocking behavior.
- Loggers returned by `create_logger()` only have a QueueHandler attached.

This design is suitable for:
- FastAPI / asyncio applications
- Background workers (RabbitMQ, Celery-like systems)
- High-throughput or I/O-sensitive services

Usage
-----

Call `create_logger()` early in application startup (e.g. in `main.py` or
FastAPI lifespan startup) to configure logging:

    logger = create_logger(
        __name__,
        structured=False,
        log_file="app.log",
    )

Subsequent calls simply return non-blocking loggers that share the same
queue and listener.
"""

import atexit
import inspect
import logging
import logging.handlers
import sys
import warnings
from pathlib import Path
from queue import Queue

from pythonjsonlogger import json as jsonlogger

ROOT = Path(__file__).parent.parent.absolute()


# -------------------------------------------------------------------
# Formatter
# -------------------------------------------------------------------


class EmojiFormatter(logging.Formatter):
    """Plain-text formatter that adds an emoji based on log level."""

    EMOJIS = {
        logging.DEBUG: "🐛",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.emoji = self.EMOJIS.get(record.levelno, "")
        return super().format(record)


def _build_formatter(structured: bool) -> logging.Formatter:
    """Create the appropriate formatter for structured or plain logs."""
    if structured:
        return jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )

    return EmojiFormatter(
        fmt="%(asctime)s - %(name)s - [%(levelname)s] %(emoji)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -------------------------------------------------------------------
# Global async logging state
# -------------------------------------------------------------------

_LOG_QUEUE: Queue[logging.LogRecord] = Queue(-1)
_LISTENER: logging.handlers.QueueListener | None = None
_LOGGING_INITIALIZED = False
_STRUCTURED_ENABLED: bool | None = None
_INIT_LEVEL: int | None = None
_INIT_LOG_FILE: str | Path | None = None


def _setup_listener(
    *,
    level: int,
    structured: bool,
    log_file: str | Path | None,
) -> None:
    """Initialize the global QueueListener (idempotent)."""
    global _LISTENER, _LOGGING_INITIALIZED, _STRUCTURED_ENABLED, _INIT_LEVEL, _INIT_LOG_FILE

    if _LOGGING_INITIALIZED:
        if level != _INIT_LEVEL:
            warnings.warn(
                f"Logging already initialized with level={_INIT_LEVEL}, "
                f"requested level={level} will be ignored",
                stacklevel=2,
            )
        if structured != _STRUCTURED_ENABLED:
            warnings.warn(
                f"Logging already initialized with structured={_STRUCTURED_ENABLED}, "
                f"requested structured={structured} will be ignored",
                stacklevel=2,
            )
        if log_file != _INIT_LOG_FILE:
            warnings.warn(
                f"Logging already initialized with log_file={_INIT_LOG_FILE!r}, "
                f"requested log_file={log_file!r} will be ignored",
                stacklevel=2,
            )
        return

    formatter = _build_formatter(structured)

    handlers: list[logging.Handler] = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    handlers.append(console)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    _LISTENER = logging.handlers.QueueListener(
        _LOG_QUEUE,
        *handlers,
        respect_handler_level=True,
    )
    _LISTENER.start()

    atexit.register(_LISTENER.stop)

    _LOGGING_INITIALIZED = True
    _STRUCTURED_ENABLED = structured
    _INIT_LEVEL = level
    _INIT_LOG_FILE = log_file


# -------------------------------------------------------------------
# Public API (backward compatible)
# -------------------------------------------------------------------


def create_logger(
    name: str = "logger",
    *,
    level: int = logging.INFO,
    structured: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Create or return a non-blocking logger.

    Notes
    -----
    - Logging configuration is process-wide.
    - The first call initializes the logging system.
    - Structured logging cannot be changed after initialization.
    """
    if name == "__main__":
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        if caller:
            spec = caller.f_globals.get("__spec__")
            if spec and getattr(spec, "name", None):
                name = spec.name

    _setup_listener(
        level=level,
        structured=structured,
        log_file=log_file,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
        logger.addHandler(logging.handlers.QueueHandler(_LOG_QUEUE))
        logger.propagate = False

    return logger


__all__ = ["ROOT", "create_logger"]
