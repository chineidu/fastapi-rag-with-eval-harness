"""Application configuration dataclasses backed by an OmegaConf YAML file."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src import ROOT
from src.schemas.models import AppConfig

config_path: Path = ROOT / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
resolved_cfg = OmegaConf.to_container(config, resolve=True)
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore


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
    return AppConfig(**dict(resolved))  # type: ignore
