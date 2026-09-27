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

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using faster-whisper (CTranslate2)."""

    name: str = "faster-whisper"
    display_name: str = "Faster Whisper (CTranslate2 Python)"

    def __init__(self, device: str = "cpu", compute_type: str = "int8"):
        self.device = device
        self.compute_type = compute_type
        self._models: dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    def _get_model(self, model_name: str):
        clean_name = model_name.replace("ggml-", "").replace(".bin", "")
        if clean_name not in self._models:
            from faster_whisper import WhisperModel
            self._models[clean_name] = WhisperModel(clean_name, device=self.device, compute_type=self.compute_type)
        return self._models[clean_name]

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

        model = self._get_model(model_name)
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
