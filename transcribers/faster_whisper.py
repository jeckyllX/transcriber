"""Faster-Whisper Python transcriber adapter with in-memory streaming and VAD."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
try:
    import numpy as np
except ImportError:
    np = None

from transcribers.base import BaseTranscriber, Segment, TranscriptionResult, Word
from transcribers.model_pool import ManagedModelPool

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using faster-whisper (CTranslate2)."""

    name: str = "faster-whisper"
    display_name: str = "Faster Whisper (CTranslate2 Python)"

    def __init__(self, device: str = "cpu", compute_type: str = "int8"):
        self.device = device
        self.compute_type = compute_type
        self.pool = ManagedModelPool[Any](
            loader=self._load_whisper_model,
            name="faster-whisper",
        )

    def is_available(self) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    def _load_whisper_model(self, clean_name: str) -> Any:
        from faster_whisper import WhisperModel
        logger.info(
            "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
            clean_name, self.device, self.compute_type,
        )
        return WhisperModel(clean_name, device=self.device, compute_type=self.compute_type)

    @property
    def _models(self) -> dict[str, Any]:
        """Backward-compatible mapping of model names to active instances."""
        return self.pool.get_active_instances()

    def _get_model(self, model_name: str) -> Any:
        """Backward-compatible model retrieval."""
        return self.pool.get_or_load(model_name)

    def evict_idle_models(self, timeout_seconds: float | None = None) -> list[str]:
        """Evict cached models that have been idle for >= timeout_seconds.
        
        Guarantees models with active processing leases are never evicted.
        """
        if timeout_seconds is None:
            from config import settings
            timeout_seconds = getattr(settings, "model_idle_timeout_seconds", 1800)
        return self.pool.evict_idle(timeout_seconds)

    def unload_all_idle(self) -> list[str]:
        """Unload all idle cached models."""
        return self.pool.unload_idle()

    def unload_model(self, model_name: str) -> bool:
        """Unload specific model if not actively in use."""
        unloaded = self.pool.unload_idle(model_name)
        return bool(unloaded)

    def get_cached_models_status(self) -> dict[str, dict[str, Any]]:
        """Return snapshot of model cache status, active leases, and idle duration."""
        return self.pool.get_status()

    def transcribe(
        self,
        audio: Path | Any,
        model_name: str = "base",
        language: str | None = None,
        vad_filter: bool = True,
        on_segment: Callable[[Segment], None] | None = None,
        word_timestamps: bool = True,
        **kwargs: Any,
    ) -> TranscriptionResult:
        if not self.is_available():
            raise RuntimeError("faster-whisper package is not installed.")

        # Accept either file path or direct in-memory NumPy float32 array
        if isinstance(audio, Path):
            audio_path = audio.resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_input = str(audio_path)
        else:
            audio_input = audio

        with self.pool.lease(model_name) as model:
            lang_arg = None if (not language or language.lower() in ("auto", "")) else language.lower()

            vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=200) if vad_filter else None

            raw_segments, info = model.transcribe(
                audio_input,
                language=lang_arg,
                beam_size=5,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
                word_timestamps=word_timestamps,
            )

            segments: list[Segment] = []
            text_parts: list[str] = []

            for i, s in enumerate(raw_segments):
                clean_text = s.text.strip()
                if clean_text:
                    word_list: list[Word] = []
                    if hasattr(s, "words") and s.words:
                        for w in s.words:
                            w_text = w.word.strip()
                            if w_text:
                                word_list.append(Word(
                                    word=w_text,
                                    start=round(w.start, 3),
                                    end=round(w.end, 3),
                                    probability=round(w.probability, 3) if hasattr(w, "probability") else None,
                                ))

                    seg = Segment(
                        id=i,
                        start=round(s.start, 2),
                        end=round(s.end, 2),
                        text=clean_text,
                        confidence=round(s.avg_logprob, 3) if hasattr(s, "avg_logprob") else None,
                        words=word_list,
                    )
                    segments.append(seg)
                    text_parts.append(clean_text)

                    if on_segment:
                        try:
                            on_segment(seg)
                        except Exception:
                            pass

            full_text = " ".join(text_parts)
            detected_lang = getattr(info, "language", language or "auto")
            duration = getattr(info, "duration", segments[-1].end if segments else 0.0)

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                language=detected_lang,
                duration=round(duration, 2),
            )

