"""Liveness and readiness probes."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, status

from src import create_logger
from src.api.core.dependencies import get_retriever
from src.api.core.response import MsgSpecJSONResponse
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.api import HealthResponse, ReadinessResponse
from src.schemas.retrieval import CollectionInfo

logger = create_logger(__name__)
router = APIRouter(tags=["health"], default_response_class=MsgSpecJSONResponse)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report process liveness; static values from api_config, no I/O."""
    api = app_config.api_config
    return HealthResponse(name=api.name, status=api.status, version=api.version)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Index not ready"}},
)
async def ready(
    retriever: Annotated[LocalRetriever, Depends(get_retriever)],
) -> ReadinessResponse | MsgSpecJSONResponse:
    """Report index readiness; 503 JSON body when the collection is missing or empty."""
    try:
        # Offload the blocking store call so slow backends never stall the loop.
        info: CollectionInfo = await asyncio.to_thread(retriever.index_info)
    except Exception:
        logger.warning("Readiness probe failed", exc_info=True)
        return MsgSpecJSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "collection": None, "chunkCount": 0},
        )
    if info.chunk_count == 0:
        return MsgSpecJSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "collection": info.collection, "chunkCount": 0},
        )
    return ReadinessResponse(
        ready=True, collection=info.collection, chunk_count=info.chunk_count
    )
