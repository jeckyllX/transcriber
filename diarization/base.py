"""Base interfaces and data structures for speaker diarization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
try:
    import numpy as np
except ImportError:
    np = None
from pydantic import BaseModel, Field


class SpeakerInterval(BaseModel):
    """A continuous audio time interval attributed to a specific speaker."""
    start: float
    end: float
    speaker: str
    confidence: float | None = None


class DiarizationResult(BaseModel):
    """Complete speaker diarization result containing segmented time intervals."""
    num_speakers: int = 0
    intervals: list[SpeakerInterval] = Field(default_factory=list)

    @property
    def speaker_names(self) -> list[str]:
        """Unique sorted speaker identifiers detected in the audio."""
        return sorted({interval.speaker for interval in self.intervals})


class BaseDiarizer(ABC):
    """Abstract base class for all speaker diarization engines."""

    name: str = "base"
    display_name: str = "Base Diarizer"

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether runtime dependencies and required models are installed."""
        pass

    @abstractmethod
    def diarize(
        self,
        audio: Path | Any,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        **kwargs: Any,
    ) -> DiarizationResult:
        """Perform speaker diarization on 16kHz mono audio (Path or float32 NumPy array)."""
        pass
