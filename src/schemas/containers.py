"""Configuration dataclasses for the application and eval pipeline."""

from dataclasses import dataclass, field

from src.schemas.types import (
    DEFAULT_RERANK_MODEL_ID,
    ChunkStrategyEnum,
    VectorStoreBackendEnum,
)


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
    timeout: int = field(
        default=120, metadata={"description": "API request timeout in seconds."}
    )
    middleware: Middleware = field(
        metadata={"description": "Middleware configuration."}
    )

    def __post_init__(self) -> None:
        """Reject non-positive request timeouts at the boundary."""
        if self.timeout <= 0:
            raise ValueError(f"API timeout must be positive, got {self.timeout}.")


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
        metadata={
            "description": "Active embeddings provider: 'local', 'api', or 'stub'."
        },
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
class QdrantConfig:
    """Connection and collection settings for the self-hosted Qdrant backend."""

    host: str = field(
        default="localhost", metadata={"description": "Qdrant server host."}
    )
    port: int = field(default=6333, metadata={"description": "Qdrant server port."})
    collection: str = field(
        default="fastapi_docs",
        metadata={"description": "Qdrant collection name for the indexed corpus."},
    )


@dataclass(slots=True, kw_only=True)
class IndexerConfig:
    """Configuration for the document indexer (chunk, embed, upsert)."""

    backend: VectorStoreBackendEnum = field(
        default=VectorStoreBackendEnum.QDRANT,
        metadata={"description": "Active vector store backend."},
    )
    qdrant: QdrantConfig = field(
        default_factory=QdrantConfig,
        metadata={"description": "Qdrant backend settings."},
    )
    chunk_size: int = field(
        default=2000, metadata={"description": "Chunk size in characters."}
    )
    overlap: int = field(
        default=0, metadata={"description": "Chunk overlap in characters."}
    )
    chunk_strategy: ChunkStrategyEnum = field(
        default=ChunkStrategyEnum.NAIVE,
        metadata={"description": "Chunking strategy: naive or structural."},
    )


@dataclass(slots=True, kw_only=True)
class RetrieverConfig:
    """Query-time retrieval configuration for the baseline adapter."""

    overfetch_factor: int = field(
        default=5,
        metadata={
            "description": (
                "Multiplier on k for the chunk-level candidate window before "
                "deduplicating to documents (must be >= 1). See ADR-0021."
            )
        },
    )
    hybrid_enabled: bool = field(
        default=False,
        metadata={"description": "Fuse Qdrant dense hits with tantivy BM25 via RRF."},
    )
    sparse_k: int = field(
        default=50,
        metadata={"description": "BM25 candidate window per query."},
    )
    rrf_k: int = field(
        default=60,
        metadata={"description": "RRF smoothing constant for dense/sparse fusion."},
    )
    dense_weight: float = field(
        default=1.0,
        metadata={
            "description": (
                "RRF weight on dense rank contributions, normalized with "
                "sparse_weight to sum to 1 so only the ratio matters "
                "(must be positive)."
            )
        },
    )
    sparse_weight: float = field(
        default=1.0,
        metadata={
            "description": (
                "RRF weight on sparse rank contributions, normalized with "
                "dense_weight to sum to 1 so only the ratio matters "
                "(must be positive)."
            )
        },
    )
    tantivy_index_dir: str = field(
        default="data/.rag-index/tantivy",
        metadata={"description": "Persisted tantivy index directory."},
    )
    rerank_enabled: bool = field(
        default=False,
        metadata={"description": "Rerank fused chunk hits before dedupe."},
    )
    rerank_model_id: str = field(
        default=DEFAULT_RERANK_MODEL_ID,
        metadata={"description": "FastEmbed cross-encoder model id."},
    )
    rerank_top_n: int = field(
        default=30,
        metadata={"description": "Fused chunk candidates entering reranker."},
    )

    def __post_init__(self) -> None:
        """Reject non-positive rerank windows at the boundary."""
        if self.rerank_top_n <= 0:
            raise ValueError(f"rerank_top_n must be positive, got {self.rerank_top_n}.")


@dataclass(slots=True, kw_only=True)
class LabelingConfig:
    """Ground-truth labeling pipeline configuration."""

    corpus_md_root: str = field(
        default="docs/fastapi/docs/en/docs",
        metadata={"description": "Root directory for markdown corpus files."},
    )
    corpus_py_root: str = field(
        default="docs/fastapi/docs_src",
        metadata={"description": "Root directory for Python corpus files."},
    )
    ground_truth_output: str = field(
        default="data/ground_truth.jsonl",
        metadata={"description": "Output path for ground truth JSONL."},
    )
    top_k: int = field(
        default=30,
        metadata={"description": "Candidates per query for LLM judge."},
    )
    concurrency: int = field(
        default=3,
        metadata={"description": "Max parallel LLM judge calls."},
    )
    max_content_length: int = field(
        default=2000,
        metadata={"description": "Max characters of doc content sent to embedder."},
    )
    max_judge_content_length: int = field(
        default=4000,
        metadata={"description": "Max characters of doc content sent to LLM judge."},
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
    labeling: LabelingConfig = field(
        default_factory=LabelingConfig,
        metadata={"description": "Ground-truth labeling pipeline settings."},
    )


DEFAULT_DB = "data/.rag-eval/runs.db"
DEFAULT_GROUND_TRUTH = "data/ground_truth.jsonl"


@dataclass(slots=True, kw_only=True)
class HarnessDefaults:
    """Default CLI argument values for harness commands."""

    k: int = 10
    concurrency: int = 3


@dataclass(slots=True, kw_only=True)
class HarnessDiffThresholds:
    """Thresholds that mark a category delta as a significant regression."""

    threshold_absolute: float = 0.05
    threshold_relative: float = 5.0


@dataclass(slots=True, kw_only=True)
class HarnessConfig:
    """Resolved harness configuration (file config with CLI overrides applied)."""

    adapter: str = ""
    ground_truth: str = DEFAULT_GROUND_TRUTH
    db: str = DEFAULT_DB
    defaults: HarnessDefaults = field(default_factory=HarnessDefaults)
    diff: HarnessDiffThresholds = field(default_factory=HarnessDiffThresholds)
