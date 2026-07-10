"""Reciprocal Rank Fusion (RRF) implementation."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    ranked_lists: list[list[UUID]],
    k: int = 60,
) -> list[tuple[UUID, float]]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: Each list contains chunk IDs ordered by relevance (best first).
        k: RRF constant. Higher values reduce the impact of rank differences.

    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: defaultdict[UUID, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
