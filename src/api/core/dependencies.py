"""FastAPI dependency injection for the RAG service."""

from fastapi import Request

from src.app.adapter import LocalRetriever


async def get_retriever(request: Request) -> LocalRetriever:
    """Resolve the shared retriever from application state."""
    return request.app.state.retriever
