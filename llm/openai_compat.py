"""OpenAI-compatible LLM Provider implementation.

Universal adapter supporting Groq, OpenRouter, OpenAI, DeepSeek, Mistral AI,
vLLM, Ollama-proxy, and any endpoint following the standard /chat/completions schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx

from config import settings
from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """Universal provider for standard OpenAI-compatible REST APIs."""

    def __init__(
        self,
        provider_id: str = "openai_compat",
        display_name: str = "OpenAI-Compatible (Cloud / Proxy)",
        api_key_getter: Callable[[], str | None] | str | None = None,
        base_url_getter: Callable[[], str] | str | None = None,
        default_model_getter: Callable[[], str] | str | None = None,
        extra_headers: dict[str, str] | Callable[[], dict[str, str]] | None = None,
        curated_models: list[str] | None = None,
        # Backwards compatibility parameters
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ):
        self.provider_id = provider_id
        self.display_name = display_name
        self._api_key_getter = api_key_getter or api_key or (lambda: settings.openai_api_key)
        self._base_url_getter = base_url_getter or base_url or (lambda: settings.openai_base_url)
        self._default_model_getter = default_model_getter or default_model or (lambda: settings.openai_default_model)
        self._extra_headers = extra_headers or {}
        self.curated_models = curated_models or []

    def get_api_key(self) -> str | None:
        """Resolve current API key."""
        if callable(self._api_key_getter):
            return self._api_key_getter()
        return self._api_key_getter

    def get_base_url(self) -> str:
        """Resolve current base URL."""
        raw = self._base_url_getter() if callable(self._base_url_getter) else self._base_url_getter
        return (raw or "https://api.openai.com/v1").rstrip("/")

    def get_default_model(self) -> str:
        """Resolve current default model."""
        raw = self._default_model_getter() if callable(self._default_model_getter) else self._default_model_getter
        return raw or "gpt-4o-mini"

    def get_extra_headers(self) -> dict[str, str]:
        """Resolve extra request headers."""
        if callable(self._extra_headers):
            return self._extra_headers()
        return dict(self._extra_headers)

    def is_configured(self) -> bool:
        """Check if provider has base URL and required API credentials."""
        return bool(self.get_api_key() and self.get_base_url())

    async def list_models(self) -> list[str]:
        """Fetch available models from remote /models endpoint with fallback to curated models."""
        default_fallback = self.curated_models or [self.get_default_model()]
        if not self.is_configured():
            return default_fallback

        url = f"{self.get_base_url()}/models"
        headers = {
            "Authorization": f"Bearer {self.get_api_key()}",
            **self.get_extra_headers(),
        }

        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    if raw_ids:
                        exclude_keywords = (
                            "embedding",
                            "tts",
                            "image",
                            "veo",
                            "lyria",
                            "robotics",
                            "aqa",
                            "clip",
                            "live-translate",
                            "transcribe",
                            "computer-use",
                            "customtools",
                        )
                        normalized: list[str] = []
                        for mid in raw_ids:
                            clean_id = mid.replace("models/", "")
                            if any(kw in clean_id.lower() for kw in exclude_keywords):
                                continue
                            normalized.append(clean_id)

                        if normalized:
                            flagships = [
                                "gemini-3.8-flash",
                                "gemini-flash-latest",
                                "gemini-3.7-flash",
                                "gemini-pro-latest",
                                "gemma-4-31b-it",
                            ]
                            combined_priority = self.curated_models + flagships
                            seen: set[str] = set()
                            top: list[str] = []
                            for m in combined_priority:
                                if m in normalized and m not in seen:
                                    seen.add(m)
                                    top.append(m)
                            rest = [m for m in normalized if m not in seen]
                            return top + rest
                        return [mid.replace("models/", "") for mid in raw_ids]
                return default_fallback
        except Exception as e:
            logger.debug("Failed to query models from %s: %s", url, e)
            return default_fallback

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream generated completion tokens from {base_url}/chat/completions."""
        if not self.is_configured():
            raise RuntimeError(f"{self.display_name} requires an API Key.")

        url = f"{self.get_base_url()}/chat/completions"
        model_name = model or self.get_default_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
            **self.get_extra_headers(),
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.3),
        }

        timeout = httpx.Timeout(120.0, connect=10.0)
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code in (429, 503) and attempt < max_retries - 1:
                            err_body = await response.aread()
                            logger.warning(
                                "%s returned HTTP %d on attempt %d/%d ('%s'). Retrying in %.1fs...",
                                self.display_name,
                                response.status_code,
                                attempt + 1,
                                max_retries,
                                err_body.decode(errors="ignore")[:80],
                                1.5 * (attempt + 1),
                            )
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue

                        if response.status_code != 200:
                            err_body = await response.aread()
                            raise RuntimeError(
                                f"{self.display_name} error (HTTP {response.status_code}): {err_body.decode(errors='ignore')}"
                            )

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    token = delta.get("content", "")
                                    if token:
                                        yield token
                                except Exception:
                                    continue
                        # If streamed successfully, terminate method
                        return

            except httpx.RequestError as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"{self.display_name} connection error: {exc}")

    async def test_connection(self) -> dict[str, Any]:
        """Test authentication and connectivity to remote provider."""
        if not self.is_configured():
            return {
                "online": False,
                "base_url": self.get_base_url(),
                "error": f"{self.display_name} API Key is missing",
            }

        url = f"{self.get_base_url()}/models"
        headers = {
            "Authorization": f"Bearer {self.get_api_key()}",
            **self.get_extra_headers(),
        }

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(url, headers=headers)
                latency_ms = round((time.time() - start_time) * 1000, 1)
                return {
                    "online": resp.status_code == 200,
                    "base_url": self.get_base_url(),
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:120]}",
                }
        except Exception as e:
            return {
                "online": False,
                "base_url": self.get_base_url(),
                "error": str(e),
            }
