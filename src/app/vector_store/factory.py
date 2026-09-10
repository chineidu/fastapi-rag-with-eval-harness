"""Factory mapping the configured backend to its vector store adapter."""

from collections.abc import Callable

from src.app.vector_store.base import VectorStore
from src.app.vector_store.qdrant import QdrantVectorStore
from src.schemas.containers import IndexerConfig
from src.schemas.types import VectorStoreBackendEnum

# Registry: backend enum -> (adapter constructor, config field on IndexerConfig).
_BACKENDS: dict[VectorStoreBackendEnum, tuple[Callable[..., VectorStore], str]] = {
    VectorStoreBackendEnum.QDRANT: (QdrantVectorStore, "qdrant"),
}


def get_vector_store(cfg: IndexerConfig) -> VectorStore:
    """Build the configured vector store backend.

    Parameters
    ----------
    cfg : IndexerConfig
        Resolved indexer configuration; ``backend`` selects the adapter
        and the matching config slice is passed to its constructor.

    Returns
    -------
    VectorStore
        A backend instance satisfying the vector store protocol.

    Raises
    ------
    ValueError
        If ``cfg.backend`` is not a registered backend.

    """
    entry = _BACKENDS.get(cfg.backend)
    if entry is None:
        raise ValueError(f"Unknown vector store backend: {cfg.backend!r}")
    backend_cls, attr = entry
    return backend_cls(getattr(cfg, attr))
