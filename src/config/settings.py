import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src import create_logger
from src.schemas.types import EnvironmentEnum

logger = create_logger(__name__)


class BaseConfig(BaseSettings):
    """Application settings class containing database and other credentials."""

    # ===== API SERVER =====
    ENV: EnvironmentEnum = EnvironmentEnum.DEVELOPMENT
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    GITHUB_READ_ACCESS: SecretStr = SecretStr("")
    STACK_EXCHANGE_READ_ACCESS: SecretStr = SecretStr("")

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: SecretStr = SecretStr("")

    @field_validator("PORT", mode="before")
    @classmethod
    def parse_port_fields(cls, v: str | int) -> int:
        """Parses port fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid port value: {v}") from None

        if isinstance(v, int) and not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")

        return v


class DevelopmentConfig(BaseConfig):
    """Development environment settings."""

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    ENV: EnvironmentEnum = EnvironmentEnum.DEVELOPMENT
    WORKERS: int = 1
    RELOAD: bool = True
    DEBUG: bool = True
    LIMIT_VALUE: int = 500  # reqs/min


class SandboxConfig(BaseConfig):
    """Sandbox environment settings."""

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    ENV: EnvironmentEnum = EnvironmentEnum.SANDBOX
    WORKERS: int = 1
    RELOAD: bool = False
    DEBUG: bool = False
    LIMIT_VALUE: int = 30  # reqs/min


class ProductionConfig(BaseConfig):
    """Production environment settings."""

    model_config = SettingsConfigDict(
        env_file=str(Path(".env").absolute()),
        env_file_encoding="utf-8",
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    ENV: EnvironmentEnum = EnvironmentEnum.PRODUCTION
    WORKERS: int = 2
    RELOAD: bool = False
    DEBUG: bool = False
    LIMIT_VALUE: int = 60  # reqs/min


type ConfigType = DevelopmentConfig | ProductionConfig | SandboxConfig


def refresh_settings() -> ConfigType:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    ConfigType
        An instance of the appropriate Settings subclass based on the ENV variable.
    """
    load_dotenv(override=True)
    # Determine environment type; `development` is the default
    env_str = os.getenv("ENV", EnvironmentEnum.DEVELOPMENT.value)
    env = EnvironmentEnum(env_str)
    logger.info("Loading configuration for environment %s: ", env.value)

    configs = {
        EnvironmentEnum.DEVELOPMENT: DevelopmentConfig,
        EnvironmentEnum.PRODUCTION: ProductionConfig,
        EnvironmentEnum.SANDBOX: SandboxConfig,
    }
    config_cls: type[ConfigType] = configs.get(env, DevelopmentConfig)

    return config_cls()


app_settings: ConfigType = refresh_settings()
