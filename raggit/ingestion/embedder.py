"""Embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from raggit.api.models import EmbeddingConfig
from raggit.core.logging import get_logger

logger = get_logger("raggit.ingestion.embedder")


def collection_name_for_model(
    base: str,
    model: str,
    version: str | None,
    vector_size: int,
) -> str:
    """Build a deterministic Qdrant collection name for an embedding model."""
    import re

    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", model).strip("_").lower()
    suffix = f"{safe_model}_{vector_size}"
    if version:
        safe_version = re.sub(r"[^a-zA-Z0-9_-]+", "_", version).strip("_").lower()
        suffix = f"{safe_model}_{safe_version}_{vector_size}"
    return f"{base}_{suffix}"


class Embedder(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts into vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @property
    @abstractmethod
    def model_version(self) -> str | None:
        """Return the pinned model version, if any."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the embedding provider name."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Return the embedding dimension."""


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers embedder."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = SentenceTransformer(config.model)

    async def embed(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> list[list[float]]:
        """Encode texts into vectors."""
        # sentence-transformers is CPU-bound; run in default executor
        import asyncio

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )
        result: list[list[float]] = embeddings.tolist()
        if progress_callback is not None:
            progress_callback(len(result), len(texts))
        return result

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def model_version(self) -> str | None:
        return self.config.model_version

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    @property
    def vector_size(self) -> int:
        return self._model.get_embedding_dimension() or 384


class OpenAIEmbedder(Embedder):
    """OpenAI-compatible API embedder."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or ""
        self.model = config.model
        self._vector_size: int | None = None

    async def embed(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> list[list[float]]:
        """Call OpenAI-compatible embedding endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model,
        }

        results: list[list[float]] = []
        async with httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=120.0
        ) as client:
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i : i + self.config.batch_size]
                payload["input"] = batch
                response = await client.post("/embeddings", json=payload)
                response.raise_for_status()
                data = response.json()["data"]
                # Sort by index to preserve order
                data.sort(key=lambda item: item["index"])
                results.extend([item["embedding"] for item in data])
                if progress_callback is not None:
                    progress_callback(min(len(results), len(texts)), len(texts))

        if results and self._vector_size is None:
            self._vector_size = len(results[0])

        return results

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def model_version(self) -> str | None:
        return self.config.model_version

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            raise RuntimeError("Vector size not known until first embedding")
        return self._vector_size


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory for embedding providers."""
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(config)
    if config.provider == "openai":
        return OpenAIEmbedder(config)
    msg = f"Unknown embedding provider: {config.provider}"
    raise ValueError(msg)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
