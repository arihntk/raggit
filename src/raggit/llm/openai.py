"""OpenAI-compatible LLM provider."""

from __future__ import annotations

import httpx

from raggit.api.models import LLMConfig
from raggit.core.logging import get_logger
from raggit.llm.base import LLMProvider

logger = get_logger("raggit.llm.openai")


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or ""
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
        """Call the chat completions endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        async with httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=120.0
        ) as client:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
