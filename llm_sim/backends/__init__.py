"""LLM backend adapters (anthropic, openai, ollama, etc.)."""

from __future__ import annotations

from llm_sim.backends.base import LLMBackend, LLMResponse
from llm_sim.config import LLMConfig

__all__ = ["LLMBackend", "LLMResponse", "create_backend"]


def create_backend(config: LLMConfig) -> LLMBackend:
    """Create an LLM backend instance based on configuration.

    Args:
        config: LLM configuration section.

    Returns:
        An initialized LLMBackend instance.

    Raises:
        ValueError: If the backend name is unknown.
    """
    backend = config.backend.lower()

    if backend == "openai":
        from llm_sim.backends.openai_backend import OpenAIBackend
        return OpenAIBackend(config)

    if backend == "anthropic":
        from llm_sim.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend(config)

    if backend in ("ollama", "ollama-cloud"):
        from llm_sim.backends.ollama_backend import OllamaBackend
        return OllamaBackend(config)

    raise ValueError(
        f"Unknown LLM backend '{config.backend}'. "
        f"Valid options: openai, anthropic, ollama, ollama-cloud"
    )
