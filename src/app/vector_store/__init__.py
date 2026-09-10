"""Vector store package: backend-agnostic contract, base, and Qdrant adapter.

The indexer and retriever depend only on :class:`VectorStore` (the
protocol) and obtain a concrete instance via :func:`get_vector_store`.
"""

from src.app.vector_store.base import BaseVectorStore, VectorStore
from src.app.vector_store.factory import get_vector_store
from src.app.vector_store.qdrant import QdrantVectorStore

__all__ = ["BaseVectorStore", "QdrantVectorStore", "VectorStore", "get_vector_store"]
