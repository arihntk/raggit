"""Evaluation metrics for raggit.

All metrics operate on normalized Python primitives so they are easy to test
and compose. Retrieval metrics expect ``retrieved`` to be ordered best-first.

This module now covers every major feature of raggit:

- Retrieval: recall/precision/mrr/ndcg/hit_rate per-filter variants
- RB: BM25 vs semantic vs hybrid, RRF, reranker, threshold, parent-window, traversal
- Ingestion: parser, chunker, cleaner, dedup, PII, injection, embedder
- Safety: groundedness, refusal, audit
- System: citation quality, hallucination, latency percentiles, filter accuracy
"""

from __future__ import annotations

import math
import re
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


# ---------------------------------------------------------------------------
# Retrieval core
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Component-specific helpers (ingestion & retrieval primitives)
# ---------------------------------------------------------------------------


def text_fidelity(predicted: str, expected: str) -> float:
    """Character-level fidelity (1 - normalized edit distance approximation).

    Uses longest-common-subsequence ratio as a cheap proxy.
    """
    if not expected:
        return 1.0 if not predicted else 0.0
    # Jaccard on character 3-grams as proxy for edit distance
    def _ngrams(s: str, n: int = 3) -> set[str]:
        s = s.lower()
        return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

    a, b = _ngrams(predicted), _ngrams(expected)
    if not a or not b:
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0
    return len(a & b) / len(a | b)


def page_preservation_score(predicted_pages: Sequence[int | None], expected_pages: Sequence[int | None]) -> float:
    """Fraction of chunks where page_number matches expected."""
    if not expected_pages:
        return 1.0
    hits = sum(1 for p, e in zip(predicted_pages, expected_pages) if p == e)
    return hits / len(expected_pages)


def section_preservation_score(predicted_titles: Sequence[str | None], expected_titles: Sequence[str | None]) -> float:
    """Fraction of chunks where section_title matches expected (case-insensitive)."""
    if not expected_titles:
        return 1.0
    def _norm(v: str | None) -> str:
        return (v or "").strip().lower()
    hits = sum(1 for p, e in zip(predicted_titles, expected_titles) if _norm(p) == _norm(e))
    return hits / len(expected_titles)


def chunk_word_count_accuracy(predicted_counts: Sequence[int], expected_counts: Sequence[int], tolerance: int = 10) -> float:
    """Fraction of chunks where word count is within tolerance of expected."""
    if not expected_counts:
        return 1.0
    hits = sum(1 for p, e in zip(predicted_counts, expected_counts) if abs(p - e) <= tolerance)
    return hits / len(expected_counts)


def dedup_effectiveness(original_count: int, deduped_count: int, expected_deduped: int) -> float:
    """How close deduped count is to expected (1 = perfect).

    Returns 1 - |predicted - expected| / original.
    """
    if original_count == 0:
        return 1.0
    diff = abs(deduped_count - expected_deduped)
    return max(0.0, 1.0 - diff / original_count)


def cleaner_effectiveness(original: str, cleaned: str, expected_cleaned: str) -> float:
    """Proxy for cleaner quality via text fidelity to expected."""
    return text_fidelity(cleaned, expected_cleaned) if expected_cleaned else 1.0


def pii_metrics(predicted_spans: Sequence[str], expected_spans: Sequence[str]) -> dict[str, float]:
    """Compute PII recall/precision/F1 on span-level."""
    pred_set = {s.lower().strip() for s in predicted_spans}
    exp_set = {s.lower().strip() for s in expected_spans}
    if not exp_set and not pred_set:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    if not exp_set:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    if not pred_set:
        return {"recall": 0.0, "precision": 1.0, "f1": 0.0}
    tp = len(pred_set & exp_set)
    recall = tp / len(exp_set) if exp_set else 0.0
    precision = tp / len(pred_set) if pred_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def injection_detection_rate(detected: Sequence[bool], expected: Sequence[bool]) -> float:
    """Accuracy of injection detection (hardened vs not)."""
    if not expected:
        return 1.0
    hits = sum(1 for d, e in zip(detected, expected) if d == e)
    return hits / len(expected)


def sanitizer_keyword_recall(predicted_keywords: Sequence[str], expected_keywords: Sequence[str]) -> float:
    """Keyword recall for sanitize_query."""
    if not expected_keywords:
        return 1.0 if not predicted_keywords else 0.0
    pred_set = {k.lower() for k in predicted_keywords}
    exp_set = {k.lower() for k in expected_keywords}
    hits = len(pred_set & exp_set)
    return hits / len(exp_set)


def rrf_fusion_quality(fused_ranking: Sequence[UUID | str], relevant: Sequence[UUID | str], k: int = 10) -> float:
    """NDCG of RRF fused list (reuses ndcg)."""
    return ndcg_at_k(fused_ranking, relevant, k=k)


def reranker_gain(mrr_before: float, mrr_after: float) -> float:
    """Relative gain from reranking: (after - before) / max(before, 1e-6)."""
    if mrr_before <= 1e-6:
        return 1.0 if mrr_after > 0 else 0.0
    return (mrr_after - mrr_before) / mrr_before


def threshold_retention_rate(before_count: int, after_count: int) -> float:
    """Fraction retained after score threshold."""
    if before_count == 0:
        return 1.0
    return after_count / before_count


def parent_window_gain(recall_with: float, recall_without: float) -> float:
    """Gain from parent-window expansion."""
    if recall_without <= 1e-9:
        return 1.0 if recall_with > 0 else 0.0
    return (recall_with - recall_without) / recall_without if recall_with >= recall_without else -abs(recall_with - recall_without)


def traversal_precision(expanded_chunks: Sequence[UUID | str], relevant: Sequence[UUID | str]) -> float:
    """Precision of traversal-expanded chunks (extra chunks beyond initial hit)."""
    if not expanded_chunks:
        return 1.0
    relevant_set = _as_uuid_set(relevant)
    hits = sum(1 for c in expanded_chunks if (c if isinstance(c, UUID) else UUID(str(c))) in relevant_set)
    return hits / len(expanded_chunks)


def citation_quality(predicted_citations: Sequence[dict], expected_citations: Sequence[dict], field: str = "chunk_id") -> float:
    """Citation precision/recall proxy – exact match on chosen field."""
    if not expected_citations:
        return 1.0 if not predicted_citations else 0.0
    pred_vals = {str(c.get(field, "")) for c in predicted_citations}
    exp_vals = {str(c.get(field, "")) for c in expected_citations}
    hits = len(pred_vals & exp_vals)
    return hits / len(exp_vals)


def hallucination_rate(groundedness_scores: Sequence[bool | None]) -> float:
    """Fraction of answers that are ungrounded (hallucinated)."""
    if not groundedness_scores:
        return 0.0
    hallucinated = sum(1 for g in groundedness_scores if g is False)
    return hallucinated / len(groundedness_scores)


def filter_accuracy(predicted_ids: Sequence[UUID | str], expected_ids: Sequence[UUID | str], total_relevant: int | None = None) -> float:
    """Filter accuracy as recall on expected filtered set."""
    return recall_at_k(predicted_ids, expected_ids, k=len(expected_ids) if expected_ids else 5)


def latency_percentile(latencies_ms: Sequence[float], percentile: float = 50.0) -> float:
    """Compute latency percentile (50 = median, 95 = p95)."""
    if not latencies_ms:
        return 0.0
    sorted_vals = sorted(latencies_ms)
    idx = math.ceil(percentile / 100 * len(sorted_vals)) - 1
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def audit_coverage(logged_events: int, expected_events: int) -> float:
    """Coverage of audit logging."""
    if expected_events == 0:
        return 1.0
    return min(1.0, logged_events / expected_events)
