"""Streaming RAG query endpoint (server-sent events)."""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

import msgspec
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src import create_logger
from src.api.core.dependencies import get_retriever
from src.api.core.exceptions import (
    GenerationError,
    RequestTimeoutError,
    UpstreamUnavailableError,
    map_provider_error,
)
from src.api.core.response import MsgSpecJSONResponse
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.api import AskRequest
from src.schemas.generation import GeneratedAnswer

logger = create_logger(__name__)
router = APIRouter(tags=["ask"], default_response_class=MsgSpecJSONResponse)

_DONE_EVENT = "data: [DONE]\n\n"


def _encode_event(answer: GeneratedAnswer) -> str:
    """Frame one partial answer as a server-sent event."""
    payload: str = msgspec.json.encode(answer.model_dump(by_alias=True)).decode()
    return f"data: {payload}\n\n"


@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest,
    retriever: Annotated[LocalRetriever, Depends(get_retriever)],
) -> StreamingResponse:
    """Stream partial answers; a final [DONE] event marks completion.

    Mid-stream snapshots may carry incomplete citations; only the last
    data event before [DONE] is citation-clamped.
    """
    timeout = app_config.api_config.timeout
    # Fetch the first snapshot before responding: StreamingResponse
    # commits 200 on entry, so failures must surface here to keep
    # their status codes.
    try:
        pending: AsyncIterator[GeneratedAnswer] = retriever.astream(
            body.query, k=body.top_k
        )
        try:
            first: GeneratedAnswer = await asyncio.wait_for(
                anext(pending), timeout=timeout
            )
        except TimeoutError as exc:
            raise RequestTimeoutError(f"Stream exceeded {timeout}s timeout") from exc
    except RequestTimeoutError, GenerationError, UpstreamUnavailableError:
        raise
    except Exception as exc:
        raise map_provider_error(
            exc, logger=logger, operation="Stream", query_preview=body.query[:120]
        ) from exc

    async def event_source() -> AsyncIterator[str]:
        try:
            yield _encode_event(first)
            async for partial in pending:
                yield _encode_event(partial)
            yield _DONE_EVENT
        except Exception:
            # The status is committed; log and end the stream instead.
            logger.exception("Stream aborted for query %r", body.query[:120])

    return StreamingResponse(event_source(), media_type="text/event-stream")
