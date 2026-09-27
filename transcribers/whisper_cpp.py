"""Whisper.cpp binary transcriber adapter."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
try:
    import numpy as np
except ImportError:
    np = None

from config import settings
from transcribers.base import BaseTranscriber, Segment, TranscriptionResult, Word

logger = logging.getLogger(__name__)


class WhisperCppTranscriber(BaseTranscriber):
    """Transcriber using Georgi Gerganov's whisper.cpp standalone binary."""

    name: str = "whisper.cpp"
    display_name: str = "Whisper C++ (High Performance)"

    def __init__(self, bin_path: str | None = None, models_dir: Path | None = None):
        self.bin_path = bin_path or settings.whisper_bin
        self.models_dir = models_dir or settings.whisper_models_dir

    def is_available(self) -> bool:
        if not self.bin_path:
            return False
        p = Path(self.bin_path)
        return p.is_file() and os.access(p, os.X_OK)

    def _resolve_model_path(self, model_name: str) -> Path:
        clean_name = model_name.replace("ggml-", "").replace(".bin", "")
        candidates = [
            self.models_dir / f"ggml-{clean_name}.bin",
            self.models_dir / f"{clean_name}.bin",
            Path(model_name),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Whisper GGML model '{model_name}' not found in {self.models_dir}."
        )

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
            raise RuntimeError("whisper.cpp binary is not installed or executable.")

        model_path = self._resolve_model_path(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            if np is not None and isinstance(audio, np.ndarray):
                # Write array to temporary WAV for whisper.cpp CLI input
                import wave
                audio_path = Path(tmp_dir) / "input.wav"
                pcm_16 = (audio * 32767.0).astype(np.int16)
                with wave.open(str(audio_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm_16.tobytes())
            else:
                audio_path = audio.resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

            out_prefix = Path(tmp_dir) / "output"
            cmd = [
                str(self.bin_path),
                "-m", str(model_path),
                "-f", str(audio_path),
                "-ojf" if word_timestamps else "-oj",
                "-of", str(out_prefix),
            ]

            if vad_filter:
                vad_candidates = [
                    self.models_dir / "ggml-silero-vad.bin",
                    self.models_dir / "silero-vad.bin",
                ]
                vad_model = next((v for v in vad_candidates if v.is_file()), None)
                if vad_model:
                    cmd.extend(["--vad", "-vm", str(vad_model)])

            if language and language.lower() not in ("auto", ""):
                cmd.extend(["-l", language.lower()])
            else:
                cmd.extend(["-l", "auto"])

            env = dict(os.environ)
            bin_parent = str(Path(self.bin_path).parent)
            current_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{bin_parent}:{current_ld}" if current_ld else bin_parent

            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"whisper.cpp failed (code {proc.returncode}): {proc.stderr or proc.stdout}")

            json_file = Path(f"{out_prefix}.json")
            if json_file.is_file():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                detected_lang = data.get("result", {}).get("language", language or "auto")
                raw_segments = data.get("transcription", [])
                segments: list[Segment] = []
                full_text_parts: list[str] = []

                for i, s in enumerate(raw_segments):
                    t0 = s.get("offsets", {}).get("from", 0) / 1000.0 if "offsets" in s else s.get("from", 0) / 1000.0
                    t1 = s.get("offsets", {}).get("to", 0) / 1000.0 if "offsets" in s else s.get("to", 0) / 1000.0
                    text = s.get("text", "").strip()
                    if text:
                        words: list[Word] = []
                        raw_tokens = s.get("tokens", [])
                        for tok in raw_tokens:
                            txt = tok.get("text", "")
                            if txt.startswith("[_") or txt.startswith("<|"):
                                continue
                            offsets = tok.get("offsets", {})
                            wt0 = offsets.get("from", 0) / 1000.0
                            wt1 = offsets.get("to", 0) / 1000.0
                            p = tok.get("p")
                            p_float = round(p, 3) if p is not None else None
                            if txt.startswith(" ") or not words:
                                clean = txt.strip()
                                if clean:
                                    words.append(Word(word=clean, start=wt0, end=wt1, probability=p_float))
                            else:
                                words[-1].word += txt
                                words[-1].end = max(words[-1].end, wt1)
                                if p_float is not None and words[-1].probability is not None:
                                    words[-1].probability = round((words[-1].probability + p_float) / 2, 3)

                        seg = Segment(id=i, start=t0, end=t1, text=text, words=words)
                        segments.append(seg)
                        full_text_parts.append(text)
                        if on_segment:
                            try:
                                on_segment(seg)
                            except Exception:
                                pass

                full_text = " ".join(full_text_parts)
                duration = segments[-1].end if segments else 0.0
                return TranscriptionResult(
                    text=full_text,
                    segments=segments,
                    language=detected_lang,
                    duration=duration,
                )

            stdout_text = proc.stdout.strip()
            return TranscriptionResult(
                text=stdout_text,
                segments=[Segment(id=0, start=0.0, end=0.0, text=stdout_text)],
                language=language or "auto",
                duration=0.0,
            )
