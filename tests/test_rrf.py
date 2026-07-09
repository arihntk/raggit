"""Tests for reciprocal rank fusion."""

from uuid import UUID

from raggit.retrieval.rrf import reciprocal_rank_fusion


def test_reciprocal_rank_fusion() -> None:
    id_a = UUID("12345678-1234-5678-1234-567812345678")
    id_b = UUID("22345678-1234-5678-1234-567812345678")
    id_c = UUID("32345678-1234-5678-1234-567812345678")

    bm25 = [id_a, id_b]
    semantic = [id_b, id_c]

    fused = reciprocal_rank_fusion([bm25, semantic], k=60)
    scores = dict(fused)

    assert id_b in scores
    assert id_a in scores
    assert id_c in scores
    # id_b appears in both lists and should have the highest score
    assert scores[id_b] > scores[id_a]
    assert scores[id_b] > scores[id_c]
