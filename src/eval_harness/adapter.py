"""Retriever adapter contract between the harness and a project-specific RAG system."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class RetrievalResult:
    """Documents retrieved for one query, with optional audit metadata.

    The harness operates at the document level (file paths). Chunk-to-doc
    deduplication is the adapter's responsibility; chunk metadata goes in
    ``metadata`` for auditability.
    """

    documents: list[tuple[str, float]]
    metadata: dict[str, object] = field(default_factory=dict)


class RetrieverAdapter(Protocol):
    """Project-specific bridge from the harness to a RAG codebase.

    Implement once per project. The harness only ever depends on this
    interface, so the harness module stays copyable across repos.
    """

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Return the top-k documents for a query.

        Parameters
        ----------
        query : str
            The ``query_text`` from a ground truth record.
        k : int
            Number of documents to return.

        Returns
        -------
        RetrievalResult
            Ranked ``(doc_path, score)`` pairs plus audit metadata.

        """
        ...

    def generate(self, query: str, documents: list[str]) -> str:
        """Generate an answer from retrieved documents (deferred).

        Parameters
        ----------
        query : str
            The user question.
        documents : list[str]
            Retrieved document paths used as context.

        Returns
        -------
        str
            The generated answer. Reserved for future generation eval.

        """
        ...
