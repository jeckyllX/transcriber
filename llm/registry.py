"""LLM Provider Registry module."""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from llm.base import BaseLLMProvider
from llm.ollama import OllamaLLMProvider
from llm.openai_compat import OpenAICompatibleLLMProvider

logger = logging.getLogger(__name__)


class LLMRegistry:
    """Registry managing available local and cloud LLM providers."""

    def __init__(self):
        self._providers: dict[str, BaseLLMProvider] = {
            "ollama": OllamaLLMProvider(),
            "groq": OpenAICompatibleLLMProvider(
                provider_id="groq",
                display_name="Groq Cloud (Ultra-Fast)",
                api_key_getter=lambda: settings.groq_api_key,
                base_url_getter=lambda: settings.groq_base_url,
                default_model_getter=lambda: settings.groq_default_model,
                curated_models=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                ],
            ),
            "openrouter": OpenAICompatibleLLMProvider(
                provider_id="openrouter",
                display_name="OpenRouter (Multi-Model Cloud)",
                api_key_getter=lambda: settings.openrouter_api_key,
                base_url_getter=lambda: settings.openrouter_base_url,
                default_model_getter=lambda: settings.openrouter_default_model,
                extra_headers={
                    "HTTP-Referer": "https://github.com/jekyll86/transcriber",
                    "X-Title": "Transcriber",
                },
                curated_models=[
                    "meta-llama/llama-3.3-70b-instruct",
                    "anthropic/claude-3.5-sonnet",
                    "deepseek/deepseek-r1",
                    "openai/gpt-4o-mini",
                ],
            ),
            "openai_compat": OpenAICompatibleLLMProvider(
                provider_id="openai_compat",
                display_name="OpenAI / Custom Endpoint",
                api_key_getter=lambda: settings.openai_api_key,
                base_url_getter=lambda: settings.openai_base_url,
                default_model_getter=lambda: settings.openai_default_model,
                curated_models=[
                    "gemini-3.8-flash",
                    "gemini-flash-latest",
                    "gemini-3.7-flash",
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-4-turbo",
                ],
            ),
        }

    def register(self, provider: BaseLLMProvider) -> None:
        self._providers[provider.provider_id] = provider
        logger.info("Registered LLM Provider: %s", provider.provider_id)

    def get_provider(self, provider_id: str | None = None) -> BaseLLMProvider:
        pid = provider_id or "ollama"
        if pid not in self._providers:
            raise KeyError(f"LLM Provider '{pid}' not found. Available: {list(self._providers.keys())}")
        return self._providers[pid]

    def list_providers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.provider_id,
                "display_name": p.display_name,
                "configured": p.is_configured(),
            }
            for p in self._providers.values()
        ]


# Global LLM Registry
llm_registry = LLMRegistry()
