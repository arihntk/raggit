"""Factory for LLM providers."""

from __future__ import annotations

from raggit.api.models import LLMConfig
from raggit.llm.base import LLMProvider
from raggit.llm.ollama import OllamaProvider
from raggit.llm.openai import OpenAIProvider


class UnsupportedLLMError(Exception):
    """Raised when an LLM provider is not supported."""


def create_llm(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from configuration."""
    provider = config.provider.lower()
    if provider == "openai":
        return OpenAIProvider(config)
    if provider == "ollama":
        return OllamaProvider(config)
    msg = f"Unsupported LLM provider: {config.provider}"
    raise UnsupportedLLMError(msg)
