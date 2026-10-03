"""Transcriber factory for dynamic engine resolution."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from config import settings
from transcribers.base import BaseTranscriber
from transcribers.whisper_cpp import WhisperCppTranscriber
from transcribers.faster_whisper import FasterWhisperTranscriber
from transcribers.openai_compat import OpenAICompatibleTranscriber

logger = logging.getLogger(__name__)


class TranscriberFactory:
    """Factory and registry for transcriber engines."""

    def __init__(self):
        self._engines: Dict[str, BaseTranscriber] = {
            "whisper.cpp": WhisperCppTranscriber(),
            "faster-whisper": FasterWhisperTranscriber(),
            "groq": OpenAICompatibleTranscriber(
                name="groq",
                display_name="Groq Cloud Whisper (Ultra-Fast)",
                base_url_getter=lambda: settings.groq_base_url,
                api_key_getter=lambda: settings.groq_api_key,
                default_model=settings.groq_default_stt_model,
                supported_models=[
                    "whisper-large-v3-turbo",
                    "whisper-large-v3",
                    "distil-whisper-large-v3-en",
                ],
            ),
            "openrouter": OpenAICompatibleTranscriber(
                name="openrouter",
                display_name="OpenRouter Cloud Whisper",
                base_url_getter=lambda: settings.openrouter_base_url,
                api_key_getter=lambda: settings.openrouter_api_key,
                default_model=settings.openrouter_default_stt_model,
                supported_models=[
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v3-turbo",
                    "openai/whisper-1",
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/jeckyllX/transcriber",
                    "X-Title": "Transcriber",
                },
            ),
            "openai": OpenAICompatibleTranscriber(
                name="openai",
                display_name="OpenAI Cloud Whisper",
                base_url_getter=lambda: settings.openai_base_url,
                api_key_getter=lambda: settings.openai_api_key,
                default_model="whisper-1",
                supported_models=["whisper-1"],
            ),
        }

    def list_engines(self) -> List[Dict[str, object]]:
        return [
            {
                "id": engine_id,
                "display_name": engine.display_name,
                "available": engine.is_available(),
                "cloud": isinstance(engine, OpenAICompatibleTranscriber),
            }
            for engine_id, engine in self._engines.items()
        ]

    def get_engine(self, engine_id: str) -> Optional[BaseTranscriber]:
        """Get an engine instance by ID without fallback."""
        return self._engines.get(engine_id)

    def get_models_for_engine(self, engine_id: str) -> List[str]:
        """Return list of models supported by the specified engine."""
        if engine_id in ("faster-whisper", "whisper.cpp"):
            return ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        engine = self._engines.get(engine_id)
        if engine and hasattr(engine, "supported_models"):
            return list(engine.supported_models)
        return ["base"]

    def get_transcriber(self, preferred_engine: Optional[str] = None) -> BaseTranscriber:
        """Get preferred transcriber if available, otherwise fallback to available engine."""
        if preferred_engine and preferred_engine in self._engines:
            engine = self._engines[preferred_engine]
            if engine.is_available():
                return engine
            logger.warning("Preferred engine %s unavailable; selecting fallback.", preferred_engine)

        # Priority 1: whisper.cpp
        if self._engines["whisper.cpp"].is_available():
            return self._engines["whisper.cpp"]

        # Priority 2: faster-whisper
        if self._engines["faster-whisper"].is_available():
            return self._engines["faster-whisper"]

        # Priority 3: groq
        if self._engines["groq"].is_available():
            return self._engines["groq"]

        # Priority 4: openrouter
        if self._engines["openrouter"].is_available():
            return self._engines["openrouter"]

        # Priority 5: openai
        if self._engines["openai"].is_available():
            return self._engines["openai"]

        raise RuntimeError("No Whisper transcriber engines are available on this system.")


# Global factory instance
transcriber_factory = TranscriberFactory()
