"""Pydantic models for the application and eval pipeline."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schemas.containers import (
    APIConfig,
    DatabaseConfig,
    EmbeddingsConfig,
    EvalPipelineConfig,
    IndexerConfig,
    RAGConfig,
    RetrieverConfig,
)


class GroundTruthRecord(BaseModel):
    """One labeled eval query with its list of relevant documents.

    ``answerable`` distinguishes queries the corpus can answer (with
    ``relevant_docs`` listing the docs) from queries no doc answers
    (``relevant_docs`` empty). The two fields must be consistent:
    ``answerable=True`` requires at least one relevant doc, and
    ``answerable=False`` requires none.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_docs: list[str] = Field(default_factory=list)
    answerable: bool = True
    source: str | None = None
    title: str | None = None
    answer_text: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _check_answerable_consistency(self) -> GroundTruthRecord:
        """Enforce the answerable/relevant_docs invariant."""
        if self.answerable and not self.relevant_docs:
            raise ValueError(
                "answerable=True requires at least one relevant_docs entry"
            )
        if not self.answerable and self.relevant_docs:
            raise ValueError("answerable=False requires relevant_docs to be empty")
        return self


class AppConfig(BaseModel):
    """Application configuration with validation."""

    api_config: APIConfig = Field(description="Configuration settings for the API")
    database_config: DatabaseConfig = Field(
        description="Configuration settings for the database"
    )
    eval_pipeline_config: EvalPipelineConfig = Field(
        description="Configuration settings for the eval data pipeline"
    )
    embeddings_config: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        description="Shared text-embedding configuration",
    )
    rag_config: RAGConfig = Field(
        description="Configuration settings for the RAG/QA pipeline"
    )
    indexer_config: IndexerConfig = Field(
        default_factory=IndexerConfig,
        description="Configuration for the document indexer and vector store",
    )
    retriever_config: RetrieverConfig = Field(
        default_factory=RetrieverConfig,
        description="Configuration for query-time retrieval",
    )
