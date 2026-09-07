"""Pure retrieval metrics: recall@k, precision@k, and per-category aggregation."""

from collections.abc import Sequence

from src.schemas.harness import ScoredQuery

OVERALL_CATEGORY = "OVERALL"


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """Compute recall@k: fraction of relevant docs caught in the top k.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ranked retrieved doc paths (longest first).
    relevant : Sequence[str]
        Ground truth relevant doc paths.
    k : int
        Retrieval depth.

    Returns
    -------
    float
        ``|retrieved[:k] ∩ relevant| / |relevant|``. Returns ``0.0`` when
        ``relevant`` is empty or ``k`` is not positive.

    """
    if k <= 0 or len(relevant) == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """Compute precision@k: fraction of the top k results that are relevant.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ranked retrieved doc paths (longest first).
    relevant : Sequence[str]
        Ground truth relevant doc paths.
    k : int
        Retrieval depth.

    Returns
    -------
    float
        ``|retrieved[:k] ∩ relevant| / k``. Returns ``0.0`` when ``k`` is
        not positive.

    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def _aggregate_scores(items: Sequence[ScoredQuery]) -> dict[str, float]:
    count: int = len(items)
    if count == 0:
        return {"recall": 0.0, "precision": 0.0, "count": 0.0}
    return {
        "recall": sum(item.recall_at_k for item in items) / count,
        "precision": sum(item.precision_at_k for item in items) / count,
        "count": float(count),
    }


def aggregate_by_category(
    scores: Sequence[ScoredQuery],
) -> dict[str, dict[str, float]]:
    """Aggregate per-query scores into per-category means plus overall mean.

    Parameters
    ----------
    scores : Sequence[ScoredQuery]
        One entry per successfully evaluated query.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of category to ``{"recall": mean, "precision": mean,
        "count": n}``. Includes an ``OVERALL`` entry with the mean across
        all queries. Empty input yields only an ``OVERALL`` entry of zeros.

    """
    by_category: dict[str, list[ScoredQuery]] = {}
    for score in scores:
        by_category.setdefault(score.category, []).append(score)

    result: dict[str, dict[str, float]] = {}
    for category, items in by_category.items():
        result[category] = _aggregate_scores(items)

    result[OVERALL_CATEGORY] = _aggregate_scores(scores)
    return result
