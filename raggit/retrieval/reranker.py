"""Cross-encoder reranking of retrieval candidates."""

from __future__ import annotations

import asyncio
from uuid import UUID

from raggit.api.models import RerankerConfig
from raggit.core.logging import get_logger

logger = get_logger("raggit.retrieval.reranker")


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize reranker logits/scores to [0, 1] using min-max scaling."""
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]


class CrossEncoderReranker:
    """Rerank (query, passage) pairs with a cross-encoder model."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            msg = "sentence-transformers is required for cross-encoder reranking"
            raise ImportError(msg) from exc
        logger.info("Loading cross-encoder reranker", model=self.config.model)
        self._model = CrossEncoder(self.config.model)

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[UUID, str]],
    ) -> list[tuple[UUID, float]]:
        """Return candidates sorted by cross-encoder score descending.

        Args:
            query: User query text.
            candidates: List of (chunk_id, passage_text).
        """
        if not candidates:
            return []

        self._load()
        assert self._model is not None

        pairs = [(query, text) for _, text in candidates]
        loop = asyncio.get_running_loop()
        raw_scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs),
        )
        scores = _normalize_scores([float(s) for s in raw_scores])

        ranked = [
            (chunk_id, score)
            for (chunk_id, _), score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def create_reranker(config: RerankerConfig) -> CrossEncoderReranker | None:
    """Factory; returns None when reranking is disabled."""
    if not config.enabled:
        return None
    return CrossEncoderReranker(config)
