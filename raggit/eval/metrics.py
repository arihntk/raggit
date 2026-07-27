"""Evaluation metrics for raggit.

All metrics operate on normalized Python primitives so they are easy to test
and compose. Retrieval metrics expect ``retrieved`` to be ordered best-first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID


def _as_uuid_set(values: Sequence[UUID | str]) -> set[UUID]:
    """Normalize a sequence of UUID-like values to a set of UUIDs."""
    result: set[UUID] = set()
    for value in values:
        if isinstance(value, UUID):
            result.add(value)
        else:
            result.add(UUID(str(value)))
    return result


def recall_at_k(
    retrieved: Sequence[UUID | str],
    relevant: Sequence[UUID | str],
    k: int = 5,
) -> float:
    """Fraction of relevant items found in the top-k retrieved items."""
    if not relevant:
        return 1.0 if not retrieved else 0.0
    relevant_set = _as_uuid_set(relevant)
    retrieved_set = _as_uuid_set(retrieved[:k])
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def precision_at_k(
    retrieved: Sequence[UUID | str],
    relevant: Sequence[UUID | str],
    k: int = 5,
) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    relevant_set = _as_uuid_set(relevant)
    retrieved_set = _as_uuid_set(retrieved[:k])
    hits = len(relevant_set & retrieved_set)
    return hits / min(k, len(retrieved))


def mean_reciprocal_rank(
    retrieved: Sequence[UUID | str],
    relevant: Sequence[UUID | str],
) -> float:
    """Reciprocal rank of the first relevant item."""
    relevant_set = _as_uuid_set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        item_uuid = item if isinstance(item, UUID) else UUID(str(item))
        if item_uuid in relevant_set:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[float]) -> float:
    """Compute discounted cumulative gain."""
    return sum(
        rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)
    )


def ndcg_at_k(
    retrieved: Sequence[UUID | str],
    relevant: Sequence[UUID | str],
    k: int = 5,
) -> float:
    """Normalized discounted cumulative gain at k (binary relevance)."""
    if not relevant:
        return 1.0 if not retrieved else 0.0
    relevant_set = _as_uuid_set(relevant)
    relevances = [
        1.0 if (item if isinstance(item, UUID) else UUID(str(item))) in relevant_set else 0.0
        for item in retrieved[:k]
    ]
    ideal = [1.0] * min(len(relevant_set), k)
    dcg = _dcg(relevances)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    retrieved: Sequence[UUID | str],
    relevant: Sequence[UUID | str],
    k: int = 5,
) -> float:
    """1 if at least one relevant item is in top-k, else 0."""
    if not relevant:
        return 1.0 if not retrieved else 0.0
    relevant_set = _as_uuid_set(relevant)
    retrieved_set = _as_uuid_set(retrieved[:k])
    return 1.0 if relevant_set & retrieved_set else 0.0


def exact_match(predicted: str | None, expected: str) -> bool:
    """Case-insensitive exact match after whitespace normalization."""
    if predicted is None:
        return False
    return predicted.strip().lower() == expected.strip().lower()


def contains_answer(predicted: str | None, expected: str) -> bool:
    """True if the expected answer text appears in the prediction."""
    if predicted is None:
        return False
    return expected.strip().lower() in predicted.strip().lower()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def refusal_accuracy(predicted_refusal: bool, expected_refusal: bool) -> float:
    """1 if the refusal decision matches expectation, else 0."""
    return 1.0 if predicted_refusal == expected_refusal else 0.0
