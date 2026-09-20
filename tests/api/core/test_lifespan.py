"""Tests for application lifespan warmup."""

from types import SimpleNamespace
from typing import cast

from fastapi import FastAPI

from src.api.core.lifespan import lifespan
from src.app.adapter import LocalRetriever
from src.app.vector_store import VectorStore
from src.embeddings.stub import StubEmbedder
from src.schemas.models import AppConfig
from src.schemas.retrieval import CollectionInfo
from src.schemas.types import DEFAULT_RERANK_MODEL_ID, ChunkStrategyEnum


class _DescribeStore:
    """VectorStore stub exposing only describe()."""

    def __init__(self, info: CollectionInfo) -> None:
        """Store the canned collection info."""
        self._info = info

    def describe(self) -> CollectionInfo:
        """Return the canned collection info."""
        return self._info


def _info(chunk_count: int = 3) -> CollectionInfo:
    """Canned collection info with the given size."""
    return CollectionInfo(
        exists=True,
        collection="fake",
        model_id=None,
        dim=None,
        chunk_count=chunk_count,
    )


def _config() -> AppConfig:
    """App config stub with hybrid retrieval disabled."""
    return cast(
        AppConfig,
        SimpleNamespace(
            embeddings_config=SimpleNamespace(),
            indexer_config=SimpleNamespace(
                chunk_size=2000, overlap=0, chunk_strategy=ChunkStrategyEnum.NAIVE
            ),
            retriever_config=SimpleNamespace(
                overfetch_factor=5,
                hybrid_enabled=False,
                sparse_k=50,
                rrf_k=60,
                dense_weight=1.0,
                sparse_weight=1.0,
                tantivy_index_dir="",
                rerank_enabled=False,
                rerank_model_id=DEFAULT_RERANK_MODEL_ID,
                rerank_top_n=30,
            ),
        ),
    )


def _retriever(store: _DescribeStore) -> LocalRetriever:
    """Real retriever wired to the describe-only store stub."""
    return LocalRetriever(
        embedder=StubEmbedder(),
        store=cast(VectorStore, store),
        config=_config(),
    )


def _app_with(retriever: object) -> FastAPI:
    """FastAPI app carrying the given retriever on state."""
    app = FastAPI()
    app.state.retriever = retriever
    return app


class TestLifespan:
    async def test_ready_index_passes_through(self) -> None:
        """Startup completes when the index probe succeeds."""
        # Given
        app = _app_with(_retriever(_DescribeStore(_info())))
        # When
        async with lifespan(app):
            entered = True
        # Then
        assert entered

    async def test_failed_probe_still_starts(self) -> None:
        """A failing probe warns but never blocks startup."""

        # Given
        class _ExplodingStore(_DescribeStore):
            """Describe stub that always fails."""

            def describe(self) -> CollectionInfo:
                """Raise instead of describing the collection."""
                raise ConnectionError("qdrant unavailable")

        app = _app_with(_retriever(_ExplodingStore(_info())))
        # When
        async with lifespan(app):
            entered = True
        # Then
        assert entered

    async def test_missing_retriever_starts_cleanly(self) -> None:
        """Startup completes when no retriever is configured."""
        # Given
        app = FastAPI()
        # When
        async with lifespan(app):
            entered = True
        # Then
        assert entered
