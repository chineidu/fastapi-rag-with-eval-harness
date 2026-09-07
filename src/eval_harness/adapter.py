"""Retriever adapter contract between the harness and a project-specific RAG system."""

from typing import Protocol

from src.schemas.harness import RetrievalResult


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
