"""Base transcription interfaces and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any
try:
    import numpy as np
except ImportError:
    np = None
from pydantic import BaseModel, Field


class Word(BaseModel):
    """A timestamped word within a segment."""
    word: str
    start: float
    end: float
    probability: float | None = None
    speaker: str | None = None


class Segment(BaseModel):
    """A timestamped segment of transcription."""
    id: int
    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete structured transcription result."""
    text: str
    segments: list[Segment] = Field(default_factory=list)
    language: str = "auto"
    duration: float = 0.0

    @staticmethod
    def _format_timestamp(seconds: float, srt_format: bool = True) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        delimiter = "," if srt_format else "."
        return f"{hrs:02d}:{mins:02d}:{secs:02d}{delimiter}{millis:03d}"

    @staticmethod
    def _format_ass_timestamp(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int(round((seconds - int(seconds)) * 100))
        if centis >= 100:
            secs += 1
            centis -= 100
        return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"

    def get_speaker_turns(self) -> list[dict[str, Any]]:
        """Group consecutive segments by the same speaker into dialogue turns."""
        turns: list[dict[str, Any]] = []
        for seg in self.segments:
            speaker = seg.speaker or "Speaker 0"
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["end"] = seg.end
                turns[-1]["text"] = (turns[-1]["text"] + " " + seg.text.strip()).strip()
                turns[-1]["segments"].append(seg.model_dump())
            else:
                turns.append({
                    "speaker": speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "segments": [seg.model_dump()],
                })
        return turns

    def to_txt(self) -> str:
        has_speakers = any(seg.speaker for seg in self.segments)
        if not has_speakers:
            return self.text.strip()
        lines = []
        current_speaker = None
        for seg in self.segments:
            speaker = seg.speaker or "Speaker 0"
            if speaker != current_speaker:
                if lines:
                    lines.append("")
                lines.append(f"[{speaker}]:")
                current_speaker = speaker
            lines.append(seg.text.strip())
        return "\n".join(lines).strip()

    def to_srt(self) -> str:
        lines = []
        for i, seg in enumerate(self.segments, start=1):
            start_str = self._format_timestamp(seg.start, srt_format=True)
            end_str = self._format_timestamp(seg.end, srt_format=True)
            text = f"{seg.speaker}: {seg.text.strip()}" if seg.speaker else seg.text.strip()
            lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
        return "\n".join(lines).strip()

    def to_vtt(self) -> str:
        lines = ["WEBVTT\n"]
        for seg in self.segments:
            start_str = self._format_timestamp(seg.start, srt_format=False)
            end_str = self._format_timestamp(seg.end, srt_format=False)
            text = f"<v {seg.speaker}>{seg.text.strip()}</v>" if seg.speaker else seg.text.strip()
            lines.append(f"{start_str} --> {end_str}\n{text}\n")
        return "\n".join(lines).strip()

    def to_word_vtt(self) -> str:
        lines = ["WEBVTT\n"]
        for seg in self.segments:
            start_str = self._format_timestamp(seg.start, srt_format=False)
            end_str = self._format_timestamp(seg.end, srt_format=False)
            lines.append(f"{start_str} --> {end_str}")
            speaker_prefix = f"<v {seg.speaker}>" if seg.speaker else ""
            speaker_suffix = "</v>" if seg.speaker else ""
            if seg.words:
                word_cues = []
                for w in seg.words:
                    w_start = self._format_timestamp(w.start, srt_format=False)
                    word_cues.append(f"<{w_start}>{w.word}")
                lines.append(f"{speaker_prefix}{' '.join(word_cues)}{speaker_suffix}\n")
            else:
                lines.append(f"{speaker_prefix}{seg.text.strip()}{speaker_suffix}\n")
        return "\n".join(lines).strip()

    def to_ass(self, title: str = "Transcription") -> str:
        header = (
            "[Script Info]\n"
            f"Title: {title}\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,50,1\n"
            "Style: Karaoke,Arial,44,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,20,20,50,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        dialogues = []
        for seg in self.segments:
            start_str = self._format_ass_timestamp(seg.start)
            end_str = self._format_ass_timestamp(seg.end)
            speaker_name = seg.speaker or ""
            if seg.words:
                k_parts = []
                for w in seg.words:
                    duration_cs = max(1, int(round((w.end - w.start) * 100)))
                    k_parts.append(f"{{\\k{duration_cs}}}{w.word}")
                karaoke_text = " ".join(k_parts)
                dialogues.append(f"Dialogue: 0,{start_str},{end_str},Karaoke,{speaker_name},0,0,0,,{karaoke_text}")
            else:
                dialogues.append(f"Dialogue: 0,{start_str},{end_str},Default,{speaker_name},0,0,0,,{seg.text.strip()}")

        return header + "\n".join(dialogues) + "\n"

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert transcription result to dictionary including formatted subtitle strings."""
        data = self.model_dump()
        data["srt"] = self.to_srt()
        data["vtt"] = self.to_vtt()
        data["ass"] = self.to_ass()
        data["speaker_turns"] = self.get_speaker_turns()
        return data


class BaseTranscriber(ABC):
    """Abstract base class for all transcriber implementations."""

    name: str = "base"
    display_name: str = "Base Transcriber"

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: Path | Any,
        model_name: str = "base",
        language: str | None = None,
        vad_filter: bool = True,
        on_segment: Callable[[Segment], None] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Perform audio transcription on 16kHz mono audio file or float32 NumPy array."""
        pass
