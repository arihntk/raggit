"""Answer quality checks: refusal, groundedness, score thresholds."""

from __future__ import annotations

import re

from raggit.api.models import QueryResult, RetrievedChunk, SafetyConfig


def apply_score_threshold(
    chunks: list[RetrievedChunk],
    min_score: float | None,
    min_keep: int = 1,
) -> list[RetrievedChunk]:
    """Drop low-confidence chunks while keeping at least min_keep if any exist."""
    if min_score is None or not chunks:
        return chunks
    filtered = [c for c in chunks if c.score >= min_score]
    if filtered:
        return filtered
    # Keep the best single chunk rather than returning nothing if min_keep allows
    if min_keep > 0:
        return chunks[:min_keep]
    return []


def should_refuse(
    chunks: list[RetrievedChunk],
    safety: SafetyConfig,
) -> tuple[bool, str | None]:
    """Decide whether to refuse answering based on retrieval quality."""
    if safety.refuse_on_empty and not chunks:
        return True, "No relevant context was found in the index."

    if (
        safety.refuse_on_low_score
        and safety.min_answer_score is not None
        and chunks
        and max(c.score for c in chunks) < safety.min_answer_score
    ):
        return (
            True,
            f"Retrieved context scores are below the minimum threshold "
            f"({safety.min_answer_score}).",
        )
    return False, None


def check_groundedness(answer: str, result: QueryResult) -> bool:
    """Heuristic groundedness: answer should share content signals with context.

    Returns True if the answer appears grounded (or is an explicit refusal).
    """
    if result.refused:
        return True

    refusal_phrases = (
        "not enough information",
        "do not contain",
        "cannot answer",
        "no relevant",
        "insufficient context",
        "i don't know",
        "i do not know",
    )
    lower = answer.lower()
    if any(p in lower for p in refusal_phrases):
        return True

    if not result.chunks:
        return False

    # Token overlap between answer and concatenated context
    context = " ".join(c.chunk.cleaned_content for c in result.chunks).lower()
    answer_tokens = set(re.findall(r"[a-z0-9]{3,}", lower))
    context_tokens = set(re.findall(r"[a-z0-9]{3,}", context))
    if not answer_tokens:
        return True
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return overlap >= 0.15


REFUSAL_MESSAGE = (
    "I cannot answer this question based on the available documents. "
    "The retrieved context is empty or below the confidence threshold."
)
