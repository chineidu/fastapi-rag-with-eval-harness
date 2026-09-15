"""Retriever adapter contract between the harness and a project-specific RAG system."""

from typing import Protocol

from src.schemas.generation import GeneratedAnswer
from src.schemas.harness import RetrievalResult
from src.schemas.retrieval import SearchHit


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
            Ranked ``RetrievedDocument`` list plus audit metadata.

        """
        ...

    async def agenerate(
        self,
        query: str,
        documents: list[SearchHit] | None = None,
        k: int = 10,
    ) -> GeneratedAnswer:
        """Generate a structured answer from retrieved chunk contexts.

        Parameters
        ----------
        query : str
            The user question.
        documents : list[SearchHit] | None
            Ranked chunk hits supplying answer grounding. When ``None``,
            the adapter retrieves ``k`` documents first.
        k : int
            Documents to retrieve when ``documents`` is ``None``.

        Returns
        -------
        GeneratedAnswer
            Answer text with citations, model id, and grounding flag.

        """
        ...
