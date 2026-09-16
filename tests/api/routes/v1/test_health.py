"""Tests for liveness and readiness probes."""

from typing import cast

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.core.dependencies import get_retriever
from src.app.adapter import LocalRetriever
from src.config import app_config
from src.schemas.retrieval import CollectionInfo


class FakeRetriever:
    """Retriever stub with a configurable collection."""

    def __init__(self, chunk_count: int = 3) -> None:
        """Serve a collection of the given size."""
        self._info = CollectionInfo(
            exists=True,
            collection="fake",
            model_id=None,
            dim=None,
            chunk_count=chunk_count,
        )

    def index_info(self) -> CollectionInfo:
        """Return the canned collection info."""
        return self._info


class ExplodingRetriever(FakeRetriever):
    """Retriever stub whose index probe always fails."""

    def index_info(self) -> CollectionInfo:
        """Raise instead of describing the collection."""
        raise ConnectionError("qdrant unavailable")


def _client(retriever: FakeRetriever) -> TestClient:
    """Test client with the retriever dependency overridden."""
    app = create_app(cast(LocalRetriever, retriever))
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


def _prefix() -> str:
    """Route prefix from the bundled config."""
    return "/" + app_config.api_config.prefix.strip("/")


class TestHealth:
    def test_returns_api_identity(self) -> None:
        """Report the configured name, status, and version."""
        # Given
        api = app_config.api_config
        # When
        with _client(FakeRetriever()) as client:
            response = client.get(f"{_prefix()}/health")
        # Then
        assert response.status_code == 200
        assert response.json() == {
            "name": api.name,
            "status": api.status,
            "version": api.version,
        }


class TestReady:
    def test_ready_when_chunks_indexed(self) -> None:
        """Report ready with collection details when chunks exist."""
        # Given / When
        with _client(FakeRetriever(chunk_count=3)) as client:
            response = client.get(f"{_prefix()}/ready")
        # Then
        assert response.status_code == 200
        assert response.json() == {
            "ready": True,
            "collection": "fake",
            "chunkCount": 3,
        }

    def test_not_ready_when_collection_empty(self) -> None:
        """Answer 503 when the collection holds no chunks."""
        # Given / When
        with _client(FakeRetriever(chunk_count=0)) as client:
            response = client.get(f"{_prefix()}/ready")
        # Then
        assert response.status_code == 503
        assert response.json() == {
            "ready": False,
            "collection": "fake",
            "chunkCount": 0,
        }

    def test_not_ready_when_probe_fails(self) -> None:
        """Answer 503 when the index probe raises."""
        # Given / When
        with _client(ExplodingRetriever()) as client:
            response = client.get(f"{_prefix()}/ready")
        # Then
        assert response.status_code == 503
        assert response.json() == {
            "ready": False,
            "collection": None,
            "chunkCount": 0,
        }
