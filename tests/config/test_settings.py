from unittest import mock

import pytest

from src.config.settings import (
    BaseConfig,
    DevelopmentConfig,
    ProductionConfig,
    SandboxConfig,
    refresh_settings,
)
from src.schemas.types import EnvironmentEnum


class TestParsePortFields:
    def test_int_in_range(self) -> None:
        assert BaseConfig.parse_port_fields(8000) == 8000
        assert BaseConfig.parse_port_fields(1) == 1
        assert BaseConfig.parse_port_fields(65535) == 65535

    def test_str_to_int(self) -> None:
        assert BaseConfig.parse_port_fields("8000") == 8000
        assert BaseConfig.parse_port_fields("  8000  ") == 8000

    def test_int_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            BaseConfig.parse_port_fields(0)

    def test_int_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            BaseConfig.parse_port_fields(65536)

    def test_invalid_str_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid port value"):
            BaseConfig.parse_port_fields("not-a-number")


class TestConfigClasses:
    def test_development_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("LIMIT_VALUE", "500")
        cfg = DevelopmentConfig()
        assert cfg.ENV == EnvironmentEnum.DEVELOPMENT
        assert cfg.WORKERS == 1
        assert cfg.RELOAD is True
        assert cfg.DEBUG is True
        assert cfg.LIMIT_VALUE == 500

    def test_sandbox_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "sandbox")
        monkeypatch.setenv("LIMIT_VALUE", "30")
        cfg = SandboxConfig()
        assert cfg.ENV == EnvironmentEnum.SANDBOX
        assert cfg.WORKERS == 1
        assert cfg.RELOAD is False
        assert cfg.DEBUG is False
        assert cfg.LIMIT_VALUE == 30

    def test_production_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("LIMIT_VALUE", "60")
        monkeypatch.setenv("WORKERS", "2")
        cfg = ProductionConfig()
        assert cfg.ENV == EnvironmentEnum.PRODUCTION
        assert cfg.WORKERS == 2
        assert cfg.RELOAD is False
        assert cfg.DEBUG is False
        assert cfg.LIMIT_VALUE == 60


class TestRefreshSettings:
    @pytest.fixture(autouse=True)
    def _mock_load_dotenv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config import settings as settings_mod

        monkeypatch.setattr(settings_mod, "load_dotenv", mock.Mock())

    def test_returns_development_when_env_is_dev(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENV", "development")
        cfg = refresh_settings()
        assert isinstance(cfg, DevelopmentConfig)
        assert cfg.ENV == EnvironmentEnum.DEVELOPMENT

    def test_returns_sandbox_when_env_is_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENV", "sandbox")
        cfg = refresh_settings()
        assert isinstance(cfg, SandboxConfig)

    def test_returns_production_when_env_is_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENV", "production")
        cfg = refresh_settings()
        assert isinstance(cfg, ProductionConfig)

    def test_defaults_to_development_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ENV", raising=False)
        cfg = refresh_settings()
        assert isinstance(cfg, DevelopmentConfig)
