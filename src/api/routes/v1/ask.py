"""Non-streaming RAG query endpoint."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from src import create_logger
from src.api.core.dependencies import get_retriever
from src.api.core.exceptions import (
    BaseAPIError,
    GenerationError,
    RequestTimeoutError,
)
from src.api.core.response import MsgSpecJSONResponse
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.api import AskRequest
from src.schemas.generation import GeneratedAnswer

logger = create_logger(__name__)
router = APIRouter(tags=["ask"], default_response_class=MsgSpecJSONResponse)


@router.post("/ask", response_model=GeneratedAnswer)
async def ask(
    body: AskRequest,
    retriever: Annotated[LocalRetriever, Depends(get_retriever)],
) -> GeneratedAnswer:
    """Retrieve context and generate a grounded answer within the configured timeout."""
    timeout = app_config.api_config.timeout
    logger.debug("Ask received: top_k=%d", body.top_k)
    try:
        return await asyncio.wait_for(
            retriever.agenerate(body.query, k=body.top_k),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise RequestTimeoutError(f"Query exceeded {timeout}s timeout") from exc
    except BaseAPIError:
        raise
    except Exception as exc:
        logger.exception("Ask failed for query %r", body.query[:120])
        raise GenerationError(str(exc)) from exc
