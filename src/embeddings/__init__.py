from typing import Any

from src.config.config import EmbeddingsConfig
from src.embeddings.api import ApiEmbedder
from src.embeddings.local import LocalEmbedder
from src.embeddings.port import EmbedderPort
from src.embeddings.stub import StubEmbedder

__all__ = [
    "ApiEmbedder",
    "EmbedderPort",
    "EmbeddingsConfig",
    "LocalEmbedder",
    "StubEmbedder",
    "get_embedder",
    "make_embedder",
]


def make_embedder(provider: str, **kwargs: Any) -> EmbedderPort:
    """Build an embedder by provider name.

    Parameters
    ----------
    provider : str
        One of ``"local"``, ``"api"``, or ``"stub"``.
    **kwargs : Any
        Forwarded to the embedder constructor (e.g. ``model_id``, ``dim``).

    Returns
    -------
    EmbedderPort
        An embedder instance for the given provider.

    Raises
    ------
    ValueError
        If ``provider`` is unknown.
    """
    if provider == "local":
        return LocalEmbedder(**kwargs)
    if provider == "api":
        return ApiEmbedder(**kwargs)
    if provider == "stub":
        return StubEmbedder(**kwargs)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def get_embedder(cfg: EmbeddingsConfig) -> EmbedderPort:
    """Build the embedder selected by an ``EmbeddingsConfig``.

    Parameters
    ----------
    cfg : EmbeddingsConfig
        Resolved embeddings configuration.

    Returns
    -------
    EmbedderPort
        ``LocalEmbedder`` when ``cfg.provider == "local"``, otherwise
        ``ApiEmbedder``.

    Raises
    ------
    ValueError
        If ``cfg.provider`` is not ``"local"`` or ``"api"``.
    """
    if cfg.provider == "local":
        return LocalEmbedder(
            model_id=cfg.local_model_id,
            cache_dir=cfg.cache_dir,
            batch_size=cfg.batch_size,
        )
    if cfg.provider == "api":
        return ApiEmbedder(model_id=cfg.api_model_id, batch_size=cfg.batch_size)
    raise ValueError(f"Unknown embeddings provider: {cfg.provider!r}")
