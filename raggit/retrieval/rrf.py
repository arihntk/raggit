"""Reciprocal Rank Fusion (RRF) implementation."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    ranked_lists: list[list[UUID]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[UUID, float]]:
    """Fuse multiple ranked lists using weighted RRF.

    Args:
        ranked_lists: Each list contains chunk IDs ordered by relevance (best first).
        k: RRF constant. Higher values reduce the impact of rank differences.
        weights: Optional per-list weights (same length as ranked_lists). Defaults to 1.0.

    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        msg = f"weights length {len(weights)} != ranked_lists length {len(ranked_lists)}"
        raise ValueError(msg)

    scores: defaultdict[UUID, float] = defaultdict(float)

    for weight, ranked_list in zip(weights, ranked_lists, strict=True):
        if weight == 0:
            continue
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
