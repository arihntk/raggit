"""Unit tests for retrieval helpers."""

from uuid import uuid4

from raggit.retrieval.engine import _as_uuid, _clamp_top_k


def test_clamp_top_k_scales_with_corpus() -> None:
    # 1000 * 0.01 = 10, within [5, 50]
    assert _clamp_top_k(1000, min_k=5, max_k=50, ratio=0.01) == 10


def test_clamp_top_k_respects_min() -> None:
    assert _clamp_top_k(10, min_k=5, max_k=50, ratio=0.01) == 5


def test_clamp_top_k_respects_max() -> None:
    assert _clamp_top_k(100_000, min_k=5, max_k=50, ratio=0.01) == 50


def test_clamp_top_k_empty_corpus() -> None:
    assert _clamp_top_k(0, min_k=5, max_k=50, ratio=0.01) == 5


def test_as_uuid_accepts_str_and_uuid() -> None:
    value = uuid4()
    assert _as_uuid(value) == value
    assert _as_uuid(str(value)) == value
