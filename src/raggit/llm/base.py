"""Abstract LLM provider."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract large language model provider."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt."""
