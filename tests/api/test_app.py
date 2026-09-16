"""Tests for the FastAPI application factory."""

from types import SimpleNamespace
from typing import cast

from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.app import create_app
from src.api.core.exceptions import BaseAPIError
from src.api.core.middleware import RequestIDMiddleware
from src.app.adapter import LocalRetriever
from src.config import app_config


def _retriever() -> LocalRetriever:
    """Duck-typed factory input without touching real backends."""
    return cast(LocalRetriever, SimpleNamespace())


class TestCreateApp:
    def test_registers_v1_routes_under_configured_prefix(self) -> None:
        """Serve health, ready, and ask under the configured prefix."""
        # Given
        prefix = "/" + app_config.api_config.prefix.strip("/")
        # When
        with TestClient(create_app(_retriever())) as client:
            health = client.get(f"{prefix}/health")
            ready = client.get(f"{prefix}/ready")
            ask = client.post(f"{prefix}/ask", json={"query": "hi"})
            root = client.get("/health")
        # Then: the bare retriever only supports liveness; the others
        # resolve and fail downstream, which still proves the mount.
        assert health.status_code == 200
        assert ready.status_code == 503
        assert ask.status_code == 500
        assert root.status_code == 404

    def test_stores_retriever_on_state(self) -> None:
        """Share the retriever with dependencies via application state."""
        # Given
        retriever = _retriever()
        # When
        app = create_app(retriever)
        # Then
        assert app.state.retriever is retriever

    def test_registers_middleware_and_handlers(self) -> None:
        """Wire request-ID, CORS, and the envelope handlers."""
        # Given / When
        app = create_app(_retriever())
        # Then
        assert any(m.cls is RequestIDMiddleware for m in app.user_middleware)
        assert any(m.cls is CORSMiddleware for m in app.user_middleware)
        assert BaseAPIError in app.exception_handlers
        assert StarletteHTTPException in app.exception_handlers
        assert RequestValidationError in app.exception_handlers
        assert Exception in app.exception_handlers
