"""Ollama LLM Provider implementation.

Connects to Ollama REST API for listing models, pulling models, and streaming generation.
"""

import json
import logging
from typing import AsyncGenerator, Any
import httpx

from config import settings
from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama AI provider for local or remote LLM execution."""

    provider_id: str = "ollama"
    display_name: str = "Ollama (Local / Self-Hosted)"

    def __init__(self, base_url: str | None = None):
        self._custom_base_url = base_url.rstrip("/") if base_url else None

    @property
    def base_url(self) -> str:
        return (self._custom_base_url or settings.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")

    @base_url.setter
    def base_url(self, value: str | None) -> None:
        self._custom_base_url = value.rstrip("/") if value else None

    def is_configured(self) -> bool:
        return bool(self.base_url)

    async def list_models(self) -> list[str]:
        """Fetch list of models currently installed in Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    return models
                return []
        except Exception as e:
            logger.debug("Ollama is currently unreachable at %s: %s", self.base_url, e)
            return []

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream generation tokens from Ollama /api/generate."""
        url = f"{self.base_url}/api/generate"
        model_name = model or settings.default_ollama_model

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        timeout = httpx.Timeout(120.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    raise RuntimeError(f"Ollama error (HTTP {response.status_code}): {err_body.decode('utf-8', errors='ignore')}")

                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

    async def pull_model_stream(self, model_name: str) -> AsyncGenerator[dict[str, Any], None]:
        """Pull a model into Ollama and yield status progress updates."""
        url = f"{self.base_url}/api/pull"
        timeout = httpx.Timeout(600.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json={"name": model_name, "stream": True}) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    yield {"status": "error", "message": f"HTTP {response.status_code}: {err_body.decode()}"}
                    return

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            continue

    async def test_connection(self) -> dict[str, Any]:
        """Test connectivity and return installed models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return {
                        "online": True,
                        "base_url": self.base_url,
                        "model_count": len(models),
                        "models": [m.get("name") for m in models],
                    }
                return {
                    "online": False,
                    "base_url": self.base_url,
                    "error": f"HTTP status {resp.status_code}",
                }
        except Exception as e:
            return {
                "online": False,
                "base_url": self.base_url,
                "error": str(e),
            }
