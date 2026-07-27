"""Tests for raggit evaluation metrics."""

from __future__ import annotations

from uuid import UUID

import pytest

from raggit.eval.metrics import (
    contains_answer,
    cosine_similarity,
    exact_match,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    refusal_accuracy,
)


@pytest.fixture
def relevant() -> list[UUID]:
    return [UUID("11111111-1111-1111-1111-111111111111")]


@pytest.fixture
def retrieved() -> list[UUID]:
    return [
        UUID("22222222-2222-2222-2222-222222222222"),
        UUID("11111111-1111-1111-1111-111111111111"),
        UUID("33333333-3333-3333-3333-333333333333"),
    ]


def test_recall_at_k(retrieved: list[UUID], relevant: list[UUID]) -> None:
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=2) == 1.0


def test_precision_at_k(retrieved: list[UUID], relevant: list[UUID]) -> None:
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=1) == 0.0
    assert precision_at_k(retrieved, relevant, k=5) == 1 / 3


def test_mean_reciprocal_rank(retrieved: list[UUID], relevant: list[UUID]) -> None:
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5


def test_hit_rate_at_k(retrieved: list[UUID], relevant: list[UUID]) -> None:
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1.0


def test_ndcg_at_k(retrieved: list[UUID], relevant: list[UUID]) -> None:
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.6309, rel=1e-2)


def test_exact_match() -> None:
    assert exact_match("Answer", "answer") is True
    assert exact_match("Answer ", "answer") is True
    assert exact_match("Different", "answer") is False
    assert exact_match(None, "answer") is False


def test_contains_answer() -> None:
    assert contains_answer("The answer is here", "answer") is True
    assert contains_answer("No match", "answer") is False
    assert contains_answer(None, "answer") is False


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_refusal_accuracy() -> None:
    assert refusal_accuracy(True, True) == 1.0
    assert refusal_accuracy(False, False) == 1.0
    assert refusal_accuracy(True, False) == 0.0
    assert refusal_accuracy(False, True) == 0.0
