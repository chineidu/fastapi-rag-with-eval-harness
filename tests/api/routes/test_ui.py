"""Tests for the bundled chat page route."""

from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.app.adapter import LocalRetriever
from src.config import app_config


def _retriever() -> LocalRetriever:
    """Duck-typed factory input without touching real backends."""
    return cast(LocalRetriever, SimpleNamespace())


class TestUiIndex:
    def test_serves_chat_page_at_root(self) -> None:
        """Serve the bundled HTML page at the unversioned root."""
        # Given / When
        with TestClient(create_app(_retriever())) as client:
            response = client.get("/")
        # Then
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_page_targets_configured_stream_endpoint(self) -> None:
        """Keep the page's fetch target in sync with the API prefix."""
        # Given
        prefix = "/" + app_config.api_config.prefix.strip("/")
        # When
        with TestClient(create_app(_retriever())) as client:
            body = client.get("/").text
        # Then
        assert f"{prefix}/ask/stream" in body

    def test_root_is_excluded_from_openapi_schema(self) -> None:
        """Keep the UI out of the versioned API contract."""
        # Given / When
        with TestClient(create_app(_retriever())) as client:
            paths = client.get("/openapi.json").json()["paths"]
        # Then
        assert "/" not in paths
