"""Embedder factory and public embedder classes."""

from typing import Any

from src.config.config import EmbeddingsConfig
from src.embeddings.api import ApiEmbedder
from src.embeddings.base import AbstractEmbedder
from src.embeddings.local import LocalEmbedder
from src.embeddings.stub import StubEmbedder
from src.schemas.types import EmbeddingProviderEnum

__all__: list[str] = [
    "AbstractEmbedder",
    "ApiEmbedder",
    "EmbeddingsConfig",
    "LocalEmbedder",
    "StubEmbedder",
    "get_embedder",
    "make_embedder",
]


def make_embedder(provider: EmbeddingProviderEnum, **kwargs: Any) -> AbstractEmbedder:
    """Build an embedder by provider name.

    Parameters
    ----------
    provider : EmbeddingProviderEnum
        One of ``"local"``, ``"api"``, or ``"stub"``.
    **kwargs : Any
        Forwarded to the embedder constructor (e.g. ``model_id``, ``dim``).

    Returns
    -------
    AbstractEmbedder
        An embedder instance for the given provider.

    Raises
    ------
    ValueError
        If ``provider`` is unknown.

    """
    if provider == EmbeddingProviderEnum.LOCAL:
        return LocalEmbedder(**kwargs)
    if provider == EmbeddingProviderEnum.API:
        return ApiEmbedder(**kwargs)
    if provider == EmbeddingProviderEnum.STUB:
        return StubEmbedder(**kwargs)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def get_embedder(cfg: EmbeddingsConfig) -> AbstractEmbedder:
    """Build the embedder selected by an ``EmbeddingsConfig``.

    Parameters
    ----------
    cfg : EmbeddingsConfig
        Resolved embeddings configuration.

    Returns
    -------
    AbstractEmbedder
        ``LocalEmbedder`` when ``cfg.provider == "local"``, ``ApiEmbedder``
        when ``cfg.provider == "api"``, or ``StubEmbedder`` when
        ``cfg.provider == "stub"``.

    Raises
    ------
    ValueError
        If ``cfg.provider`` is not ``"local"``, ``"api"``, or ``"stub"``.

    """
    if cfg.provider == EmbeddingProviderEnum.LOCAL:
        return LocalEmbedder(
            model_id=cfg.local_model_id,
            cache_dir=cfg.cache_dir,
            batch_size=cfg.batch_size,
        )
    if cfg.provider == EmbeddingProviderEnum.API:
        return ApiEmbedder(model_id=cfg.api_model_id, batch_size=cfg.batch_size)
    if cfg.provider == EmbeddingProviderEnum.STUB:
        return StubEmbedder()
    raise ValueError(f"Unknown embeddings provider: {cfg.provider!r}")
