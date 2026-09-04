"""Application configuration dataclasses backed by an OmegaConf YAML file."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src import ROOT
from src.schemas.models import AppConfig

config_path: Path = ROOT / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
resolved_cfg = OmegaConf.to_container(config, resolve=True)
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
