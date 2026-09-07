"""OmegaConf-backed harness configuration (``.rag-eval.yaml``)."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.schemas.containers import (
    DEFAULT_DB,
    DEFAULT_GROUND_TRUTH,
    HarnessConfig,
    HarnessDefaults,
    HarnessDiffThresholds,
)

DEFAULT_CONFIG_PATH = Path(".rag-eval.yaml")


def _validate_config(cfg: HarnessConfig) -> HarnessConfig:
    """Check range invariants shared by file-loaded and CLI-overridden configs.

    Parameters
    ----------
    cfg : HarnessConfig
        Config to validate.

    Returns
    -------
    HarnessConfig
        The same config, unchanged.

    Raises
    ------
    ValueError
        If ``k``, ``concurrency``, or thresholds are out of range.

    """
    if cfg.defaults.k <= 0:
        raise ValueError(f"k must be positive, got {cfg.defaults.k}")
    if cfg.defaults.concurrency <= 0:
        raise ValueError(
            f"concurrency must be positive, got {cfg.defaults.concurrency}"
        )
    if cfg.diff.threshold_absolute < 0:
        raise ValueError("threshold_absolute must not be negative")
    if cfg.diff.threshold_relative < 0:
        raise ValueError("threshold_relative must not be negative")
    return cfg


def _from_dict(data: dict[str, Any]) -> HarnessConfig:
    """Build a validated ``HarnessConfig`` from a plain dict.

    Parameters
    ----------
    data : dict[str, Any]
        Raw config mapping (e.g. from ``OmegaConf.to_container``).

    Returns
    -------
    HarnessConfig
        Validated config.

    Raises
    ------
    ValueError
        If ``k``, ``concurrency``, or thresholds are out of range.

    """
    defaults_raw = data.get("defaults", {}) or {}
    diff_raw = data.get("diff", {}) or {}
    cfg = HarnessConfig(
        adapter=str(data.get("adapter", "")),
        ground_truth=str(data.get("ground_truth", DEFAULT_GROUND_TRUTH)),
        db=str(data.get("db", DEFAULT_DB)),
        defaults=HarnessDefaults(
            k=int(defaults_raw.get("k", 10)),
            concurrency=int(defaults_raw.get("concurrency", 3)),
        ),
        diff=HarnessDiffThresholds(
            threshold_absolute=float(diff_raw.get("threshold_absolute", 0.05)),
            threshold_relative=float(diff_raw.get("threshold_relative", 5.0)),
        ),
    )
    return _validate_config(cfg)


def load_harness_config(path: Path | str | None = None) -> HarnessConfig:
    """Load harness config from YAML, falling back to defaults when missing.

    Parameters
    ----------
    path : Path | str | None
        Explicit config path. When ``None``, ``.rag-eval.yaml`` in the
        current directory is used if it exists.

    Returns
    -------
    HarnessConfig
        Merged file config over built-in defaults.

    """
    candidate = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    base = OmegaConf.structured(HarnessConfig)
    if not candidate.exists():
        merged = base
    else:
        file_cfg = OmegaConf.load(str(candidate))
        merged = OmegaConf.merge(base, file_cfg)
    raw = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Invalid harness config in {candidate}")
    data: dict[str, Any] = {str(key): value for key, value in raw.items()}
    return _from_dict(data)


def apply_cli_overrides(
    cfg: HarnessConfig,
    *,
    adapter: str | None = None,
    ground_truth: str | None = None,
    db: str | None = None,
    k: int | None = None,
    concurrency: int | None = None,
    threshold_absolute: float | None = None,
    threshold_relative: float | None = None,
) -> HarnessConfig:
    """Apply CLI values over a file config (CLI wins when not ``None``).

    Parameters
    ----------
    cfg : HarnessConfig
        File-loaded config.
    adapter : str | None
        Adapter import path override.
    ground_truth : str | None
        Ground truth path override.
    db : str | None
        SQLite path override.
    k : int | None
        Retrieval depth override.
    concurrency : int | None
        Concurrency override.
    threshold_absolute : float | None
        Absolute diff threshold override.
    threshold_relative : float | None
        Relative diff threshold override (percent).

    Returns
    -------
    HarnessConfig
        New config with overrides applied.

    """
    return _validate_config(
        HarnessConfig(
            adapter=adapter if adapter is not None else cfg.adapter,
            ground_truth=ground_truth if ground_truth is not None else cfg.ground_truth,
            db=db if db is not None else cfg.db,
            defaults=HarnessDefaults(
                k=k if k is not None else cfg.defaults.k,
                concurrency=concurrency
                if concurrency is not None
                else cfg.defaults.concurrency,
            ),
            diff=HarnessDiffThresholds(
                threshold_absolute=(
                    threshold_absolute
                    if threshold_absolute is not None
                    else cfg.diff.threshold_absolute
                ),
                threshold_relative=(
                    threshold_relative
                    if threshold_relative is not None
                    else cfg.diff.threshold_relative
                ),
            ),
        )
    )
