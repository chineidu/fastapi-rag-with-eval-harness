import pytest
from pydantic import ValidationError

from src.config.config import (
    CORS,
    APIConfig,
    AppConfig,
    DatabaseConfig,
    Middleware,
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
        }
        cfg = AppConfig(**data)
        assert cfg.api_config.title == "API"
        assert cfg.database_config.pool_size == 30

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
