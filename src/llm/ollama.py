"""Ollama local LLM provider."""

from __future__ import annotations

import httpx
from raggit.api.models import LLMConfig
from raggit.core.logging import get_logger
from raggit.llm.base import LLMProvider

logger = get_logger("raggit.llm.ollama")


class OllamaProvider(LLMProvider):
    """Provider for local Ollama chat completion API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def generate(
        self,
        system_prompt: str | None,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call Ollama chat endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
