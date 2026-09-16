"""Application lifespan: startup warmup and shutdown."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import create_logger
from src.app.adapter import LocalRetriever

logger = create_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Probe the index on startup so misconfiguration surfaces in logs, not at first query.

    The retriever is built by the caller of ``create_app`` and shared via
    ``app.state``; lifespan only probes it and never blocks startup.
    """
    retriever = getattr(app.state, "retriever", None)
    if isinstance(retriever, LocalRetriever):
        try:
            # Offload the blocking store call so slow backends never hang startup.
            info = await asyncio.to_thread(retriever.index_info)
            logger.info(
                "Index ready: collection=%s chunks=%d",
                info.collection,
                info.chunk_count,
            )
        except Exception:
            logger.warning(
                "Index probe failed; /ready will report not-ready", exc_info=True
            )
    yield
