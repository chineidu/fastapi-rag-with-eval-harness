from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from src import ROOT


@dataclass(slots=True, kw_only=True)
class CORS:
    """CORS configuration class."""

    allow_origins: list[str] = field(
        default_factory=list, metadata={"description": "Allowed origins for CORS."}
    )
    allow_credentials: bool = field(
        metadata={"description": "Allow credentials for CORS."}
    )
    allow_methods: list[str] = field(
        default_factory=list, metadata={"description": "Allowed methods for CORS."}
    )
    allow_headers: list[str] = field(
        default_factory=list, metadata={"description": "Allowed headers for CORS."}
    )


@dataclass(slots=True, kw_only=True)
class Middleware:
    """Middleware configuration class."""

    cors: CORS = field(metadata={"description": "CORS configuration."})


@dataclass(slots=True, kw_only=True)
class APIConfig:
    """API-level configuration."""

    title: str = field(metadata={"description": "The title of the API."})
    name: str = field(metadata={"description": "The name of the API."})
    description: str = field(metadata={"description": "The description of the API."})
    version: str = field(metadata={"description": "The version of the API."})
    status: str = field(metadata={"description": "The current status of the API."})
    prefix: str = field(metadata={"description": "The prefix for the API routes."})
    middleware: Middleware = field(
        metadata={"description": "Middleware configuration."}
    )


@dataclass(slots=True, kw_only=True)
class DatabaseConfig:
    """Database configuration class."""

    pool_size: int = field(
        default=30, metadata={"description": "Number of connections to keep in pool"}
    )
    max_overflow: int = field(
        default=10, metadata={"description": "Number of extra connections allowed"}
    )
    pool_timeout: int = field(
        default=20, metadata={"description": "Seconds to wait for a connection"}
    )
    pool_recycle: int = field(
        default=1800,
        metadata={"description": "Seconds after which to recycle connections"},
    )
    pool_pre_ping: bool = field(
        default=True, metadata={"description": "Whether to test connections before use"}
    )
    expire_on_commit: bool = field(
        default=False, metadata={"description": "Whether to expire objects on commit"}
    )


@dataclass(slots=True, kw_only=True)
class GitHubEvalConfig:
    """GitHub data-fetching pipeline configuration."""

    graphql_url: str = field(metadata={"description": "GitHub GraphQL API endpoint."})
    page_size: int = field(
        default=100, metadata={"description": "Number of discussions per request."}
    )
    max_retries: int = field(
        default=3, metadata={"description": "Max retries on rate limit."}
    )
    retry_sleep_secs: float = field(
        default=0.5, metadata={"description": "Sleep between pagination requests."}
    )


@dataclass(slots=True, kw_only=True)
class StackExchangeEvalConfig:
    """Stack Exchange data-fetching pipeline configuration."""

    api_url: str = field(metadata={"description": "Stack Exchange API base URL."})
    page_size: int = field(
        default=100, metadata={"description": "Number of questions per request."}
    )
    retry_sleep_secs: float = field(
        default=1.0, metadata={"description": "Sleep between pagination requests."}
    )


@dataclass(slots=True, kw_only=True)
class ClassifierConfig:
    """LLM configuration for eval dataset classification."""

    model_id: str = field(metadata={"description": "OpenRouter model identifier."})
    max_input_length: int = field(
        default=2000, metadata={"description": "Max characters in classifier input."}
    )
    timeout_seconds: int = field(
        default=120, metadata={"description": "API request timeout in seconds."}
    )
    max_retries: int = field(
        default=3, metadata={"description": "Max retries for API requests."}
    )
    temperature: float = field(
        default=0.0, metadata={"description": "LLM sampling temperature."}
    )
    seed: int = field(
        default=47, metadata={"description": "LLM random seed for reproducibility."}
    )


@dataclass(slots=True, kw_only=True)
class EvalDefaultsConfig:
    """Default CLI argument values for eval pipeline scripts."""

    github_url: str = field(metadata={"description": "Default GitHub repo URL."})
    stackoverflow_url: str = field(
        metadata={"description": "Default Stack Exchange site URL."}
    )
    num_issues: int = field(
        default=30, metadata={"description": "Default number of items to fetch."}
    )
    github_category: str = field(
        default="questions",
        metadata={"description": "Default discussion category slug."},
    )
    stackoverflow_tag: str = field(
        default="fastapi", metadata={"description": "Default Stack Exchange tag."}
    )
    github_discussions_path: str = field(
        metadata={"description": "Default output path for GitHub discussions JSONL."}
    )
    stackoverflow_questions_path: str = field(
        metadata={
            "description": "Default output path for Stack Overflow questions JSONL."
        }
    )
    eval_dataset_path: str = field(
        metadata={"description": "Default path for unified eval dataset JSONL."}
    )
    eval_dataset_labeled_path: str = field(
        metadata={"description": "Default path for labeled eval dataset JSONL."}
    )


@dataclass(slots=True, kw_only=True)
class EmbeddingsConfig:
    """Shared text-embedding configuration for the labeling pipeline and RAG retriever."""

    provider: str = field(
        default="local",
        metadata={"description": "Active embeddings provider: 'local' or 'api'."},
    )
    local_model_id: str = field(
        default="BAAI/bge-small-en-v1.5",
        metadata={"description": "fastembed model id used by LocalEmbedder."},
    )
    api_model_id: str = field(
        default="openai/text-embedding-3-small",
        metadata={"description": "OpenRouter embeddings model id used by ApiEmbedder."},
    )
    cache_dir: str = field(
        default="data/.rag-eval/embeddings_cache",
        metadata={"description": "fastembed model download cache (gitignored)."},
    )
    batch_size: int = field(
        default=32,
        metadata={"description": "Maximum texts per embed call."},
    )


@dataclass(slots=True, kw_only=True)
class RAGLLMConfig:
    """LLM configuration for the RAG/QA pipeline."""

    model_id: str = field(metadata={"description": "OpenRouter model identifier."})
    temperature: float = field(
        default=0.1, metadata={"description": "LLM sampling temperature."}
    )
    max_tokens: int = field(
        default=4096, metadata={"description": "Max tokens in generated response."}
    )
    timeout_seconds: int = field(
        default=120, metadata={"description": "API request timeout in seconds."}
    )
    max_retries: int = field(
        default=3, metadata={"description": "Max retries for API requests."}
    )
    seed: int = field(
        default=47, metadata={"description": "LLM random seed for reproducibility."}
    )


@dataclass(slots=True, kw_only=True)
class RAGConfig:
    """RAG/QA pipeline configuration."""

    llm: RAGLLMConfig = field(
        metadata={"description": "LLM settings for RAG/QA generation."}
    )


@dataclass(slots=True, kw_only=True)
class EvalPipelineConfig:
    """Eval data pipeline configuration."""

    github: GitHubEvalConfig = field(
        metadata={"description": "GitHub fetching settings."}
    )
    stack_exchange: StackExchangeEvalConfig = field(
        metadata={"description": "Stack Exchange fetching settings."}
    )
    classifier: ClassifierConfig = field(
        metadata={"description": "LLM classification settings."}
    )
    defaults: EvalDefaultsConfig = field(
        metadata={"description": "Default CLI argument values."}
    )


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


config_path: Path = ROOT / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
resolved_cfg = OmegaConf.to_container(config, resolve=True)
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
