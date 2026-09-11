"""Application configuration dataclasses backed by an OmegaConf YAML file."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src import ROOT
from src.schemas.models import AppConfig

config_path: Path = ROOT / "src/config/config.yaml"


def load_app_config(path: str | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file.

    Parameters
    ----------
    path : str | None
        YAML file to load. Defaults to the bundled ``config.yaml`` when ``None``.

    Returns
    -------
    AppConfig
        Parsed and validated application configuration.

    """
    cfg_path = Path(path) if path else config_path
    loaded = OmegaConf.load(cfg_path).config
    resolved = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Invalid app config in {cfg_path}")
    data: dict[str, Any] = {str(key): value for key, value in resolved.items()}
    return AppConfig(**data)


app_config: AppConfig = load_app_config()
