"""Tests for API dependency injection."""

from types import SimpleNamespace

from fastapi import FastAPI, Request

from src.api.core.dependencies import get_retriever


class TestGetRetriever:
    async def test_returns_retriever_from_app_state(self) -> None:
        """Resolve the shared retriever stored on application state."""
        # Given
        app = FastAPI()
        app.state.retriever = SimpleNamespace()
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": app,
        }
        # When
        retriever = await get_retriever(Request(scope))
        # Then
        assert retriever is app.state.retriever
