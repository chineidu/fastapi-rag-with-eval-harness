"""Pydantic models for the application and eval pipeline."""

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.containers import (
    APIConfig,
    DatabaseConfig,
    EmbeddingsConfig,
    EvalPipelineConfig,
    RAGConfig,
)


class GroundTruthRecord(BaseModel):
    """One labeled eval query with its list of relevant documents."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_docs: list[str] = Field(min_length=1)
    source: str | None = None
    title: str | None = None
    answer_text: str | None = None
    url: str | None = None


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
