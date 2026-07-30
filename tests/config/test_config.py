import pytest
from pydantic import ValidationError

from src.config.config import (
    CORS,
    APIConfig,
    AppConfig,
    ClassifierConfig,
    DatabaseConfig,
    EvalDefaultsConfig,
    EvalPipelineConfig,
    GitHubEvalConfig,
    Middleware,
    RAGConfig,
    RAGLLMConfig,
    StackExchangeEvalConfig,
    app_config,
)


class TestCORS:
    def test_default_factories(self) -> None:
        cors = CORS(allow_credentials=False)
        assert cors.allow_origins == []
        assert cors.allow_methods == []
        assert cors.allow_headers == []

    def test_custom_values(self) -> None:
        cors = CORS(
            allow_origins=["http://localhost"],
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["Authorization"],
        )
        assert cors.allow_origins == ["http://localhost"]
        assert cors.allow_credentials is True
        assert cors.allow_methods == ["GET"]


class TestMiddleware:
    def test_requires_cors(self) -> None:
        cors = CORS(allow_credentials=False)
        mw = Middleware(cors=cors)
        assert mw.cors is cors


class TestAPIConfig:
    def test_requires_fields(self) -> None:
        cors = CORS(allow_credentials=False)
        mw = Middleware(cors=cors)
        cfg = APIConfig(
            title="Test API",
            name="Test",
            description="A test API",
            version="1.0.0",
            status="healthy",
            prefix="/api",
            middleware=mw,
        )
        assert cfg.title == "Test API"
        assert cfg.version == "1.0.0"


class TestDatabaseConfig:
    def test_defaults(self) -> None:
        cfg = DatabaseConfig()
        assert cfg.pool_size == 30
        assert cfg.max_overflow == 10
        assert cfg.pool_timeout == 20
        assert cfg.pool_recycle == 1800
        assert cfg.pool_pre_ping is True
        assert cfg.expire_on_commit is False

    def test_custom_values(self) -> None:
        cfg = DatabaseConfig(pool_size=10, max_overflow=5)
        assert cfg.pool_size == 10
        assert cfg.max_overflow == 5


class TestGitHubEvalConfig:
    def test_defaults(self) -> None:
        cfg = GitHubEvalConfig(graphql_url="https://api.github.com/graphql")
        assert cfg.page_size == 100
        assert cfg.max_retries == 3
        assert cfg.retry_sleep_secs == 0.5


class TestStackExchangeEvalConfig:
    def test_defaults(self) -> None:
        cfg = StackExchangeEvalConfig(api_url="https://api.stackexchange.com/2.3")
        assert cfg.page_size == 100
        assert cfg.retry_sleep_secs == 1.0


class TestClassifierConfig:
    def test_defaults(self) -> None:
        cfg = ClassifierConfig(model_id="test-model")
        assert cfg.max_input_length == 2000
        assert cfg.timeout_seconds == 120
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.0
        assert cfg.seed == 47


class TestRAGLLMConfig:
    def test_defaults(self) -> None:
        cfg = RAGLLMConfig(model_id="test-model")
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 4096
        assert cfg.timeout_seconds == 120
        assert cfg.max_retries == 3
        assert cfg.seed == 47


class TestRAGConfig:
    def test_requires_llm(self) -> None:
        llm = RAGLLMConfig(model_id="test-model")
        cfg = RAGConfig(llm=llm)
        assert cfg.llm.model_id == "test-model"


class TestEvalDefaultsConfig:
    def test_defaults(self) -> None:
        cfg = EvalDefaultsConfig(
            github_url="https://github.com/test/test",
            stackoverflow_url="https://stackoverflow.com",
            github_discussions_path="data/test.jsonl",
            stackoverflow_questions_path="data/test_so.jsonl",
            eval_dataset_path="data/test_eval.jsonl",
            eval_dataset_labeled_path="data/test_eval_labeled.jsonl",
        )
        assert cfg.num_issues == 30
        assert cfg.github_category == "questions"
        assert cfg.stackoverflow_tag == "fastapi"


class TestEvalPipelineConfig:
    def test_requires_nested_configs(self) -> None:
        github = GitHubEvalConfig(graphql_url="https://api.github.com/graphql")
        se = StackExchangeEvalConfig(api_url="https://api.stackexchange.com/2.3")
        classifier = ClassifierConfig(model_id="test-model")
        defaults = EvalDefaultsConfig(
            github_url="https://github.com/test/test",
            stackoverflow_url="https://stackoverflow.com",
            github_discussions_path="data/test.jsonl",
            stackoverflow_questions_path="data/test_so.jsonl",
            eval_dataset_path="data/test_eval.jsonl",
            eval_dataset_labeled_path="data/test_eval_labeled.jsonl",
        )
        cfg = EvalPipelineConfig(
            github=github, stack_exchange=se, classifier=classifier, defaults=defaults
        )
        assert cfg.github is github
        assert cfg.classifier.model_id == "test-model"


class TestAppConfig:
    def test_validates_good_dict(self) -> None:
        data = {
            "api_config": {
                "title": "API",
                "name": "api",
                "description": "desc",
                "version": "1.0",
                "status": "ok",
                "prefix": "/api",
                "middleware": {"cors": {"allow_credentials": False}},
            },
            "database_config": {},
            "eval_pipeline_config": {
                "github": {"graphql_url": "https://api.github.com/graphql"},
                "stack_exchange": {"api_url": "https://api.stackexchange.com/2.3"},
                "classifier": {"model_id": "test-model"},
                "defaults": {
                    "github_url": "https://github.com/test/test",
                    "stackoverflow_url": "https://stackoverflow.com",
                    "github_discussions_path": "data/test.jsonl",
                    "stackoverflow_questions_path": "data/test_so.jsonl",
                    "eval_dataset_path": "data/test_eval.jsonl",
                    "eval_dataset_labeled_path": "data/test_eval_labeled.jsonl",
                },
            },
            "rag_config": {"llm": {"model_id": "test-model"}},
        }
        cfg = AppConfig(**data)
        assert cfg.api_config.title == "API"
        assert cfg.database_config.pool_size == 30
        assert (
            cfg.eval_pipeline_config.github.graphql_url
            == "https://api.github.com/graphql"
        )

    def test_rejects_missing_rag_config(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {
                    "api_config": {
                        "title": "x",
                        "name": "x",
                        "description": "x",
                        "version": "1",
                        "status": "ok",
                        "prefix": "/",
                        "middleware": {"cors": {"allow_credentials": False}},
                    },
                    "database_config": {},
                    "eval_pipeline_config": {
                        "github": {"graphql_url": "https://api.github.com/graphql"},
                        "stack_exchange": {
                            "api_url": "https://api.stackexchange.com/2.3"
                        },
                        "classifier": {"model_id": "test-model"},
                        "defaults": {
                            "github_url": "https://github.com/test/test",
                            "stackoverflow_url": "https://stackoverflow.com",
                            "github_discussions_path": "data/test.jsonl",
                            "stackoverflow_questions_path": "data/test_so.jsonl",
                            "eval_dataset_path": "data/test_eval.jsonl",
                            "eval_dataset_labeled_path": "data/test_eval_labeled.jsonl",
                        },
                    },
                }
            )

    def test_rejects_missing_api_config(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"database_config": {}})


class TestModuleLevelAppConfig:
    def test_app_config_is_loaded(self) -> None:
        assert app_config is not None
        assert app_config.api_config.title == "RAG-based Question Answering System"
        assert app_config.api_config.prefix == "/api/v1"
        assert app_config.database_config.pool_size == 30
        assert app_config.database_config.expire_on_commit is False
        assert (
            app_config.eval_pipeline_config.github.graphql_url
            == "https://api.github.com/graphql"
        )
        assert (
            app_config.eval_pipeline_config.classifier.model_id
            == "deepseek/deepseek-v4-flash"
        )
        assert app_config.eval_pipeline_config.defaults.github_category == "questions"
        assert app_config.rag_config.llm.model_id == "deepseek/deepseek-v4-flash"
        assert app_config.rag_config.llm.temperature == 0.1
