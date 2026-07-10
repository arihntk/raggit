"""Tests for retrieval safety and quality checks."""

from datetime import UTC, datetime
from uuid import uuid4

from raggit.api.models import (
    Chunk,
    QueryResult,
    RetrievedChunk,
    SafetyConfig,
)
from raggit.retrieval.safety import (
    apply_score_threshold,
    check_groundedness,
    should_refuse,
)


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            raw_content="content",
            cleaned_content="the quick brown fox",
            created_at=datetime.now(UTC),
        ),
        score=score,
    )


def test_apply_score_threshold_filters_low_scores() -> None:
    chunks = [_chunk(0.1), _chunk(0.5), _chunk(0.9)]
    result = apply_score_threshold(chunks, 0.4, min_keep=0)
    assert len(result) == 2
    assert all(r.score >= 0.4 for r in result)


def test_apply_score_threshold_keeps_best_when_all_below() -> None:
    chunks = [_chunk(0.1), _chunk(0.05)]
    result = apply_score_threshold(chunks, 0.5, min_keep=1)
    assert len(result) == 1
    assert result[0].score == 0.1


def test_should_refuse_on_empty() -> None:
    safety = SafetyConfig(refuse_on_empty=True)
    refused, reason = should_refuse([], safety)
    assert refused is True
    assert reason is not None


def test_should_refuse_on_low_score() -> None:
    safety = SafetyConfig(refuse_on_low_score=True, min_answer_score=0.5)
    refused, reason = should_refuse([_chunk(0.1)], safety)
    assert refused is True
    assert reason is not None


def test_check_groundedness_passes_with_overlap() -> None:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        raw_content="",
        cleaned_content="the quick brown fox jumps over the lazy dog",
        created_at=datetime.now(UTC),
    )
    result = QueryResult(
        query="what does the fox do",
        sanitized_keywords=["fox"],
        chunks=[RetrievedChunk(chunk=chunk, score=1.0)],
        total_chunks_considered=1,
    )
    assert check_groundedness("The fox jumps over the lazy dog.", result) is True


def test_check_groundedness_fails_without_overlap() -> None:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        raw_content="",
        cleaned_content="the quick brown fox",
        created_at=datetime.now(UTC),
    )
    result = QueryResult(
        query="what is the weather",
        sanitized_keywords=["weather"],
        chunks=[RetrievedChunk(chunk=chunk, score=1.0)],
        total_chunks_considered=1,
    )
    assert check_groundedness("Weather precipitation atmospheric conditions vary.", result) is False
