"""Real-time streaming transcription subsystem.

Provides in-memory audio buffering, energy-based VAD filtering, and sliding-window
Whisper inference for live microphone dictation over WebSockets.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
import numpy as np

from transcribers.base import Segment, TranscriptionResult, Word
from transcribers.factory import transcriber_factory

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """In-memory audio sample buffer for streaming 16kHz mono float32 PCM."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0
        self._committed_samples: int = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def committed_samples(self) -> int:
        return self._committed_samples

    @property
    def duration_seconds(self) -> float:
        return self._total_samples / self.sample_rate

    @property
    def committed_duration_seconds(self) -> float:
        return self._committed_samples / self.sample_rate

    @property
    def uncommitted_samples_count(self) -> int:
        return max(0, self._total_samples - self._committed_samples)

    @property
    def uncommitted_duration_seconds(self) -> float:
        return self.uncommitted_samples_count / self.sample_rate

    def append_pcm16(self, pcm_bytes: bytes) -> int:
        """Decode signed 16-bit little-endian PCM bytes and append to buffer."""
        if not pcm_bytes:
            return 0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._chunks.append(samples)
        self._total_samples += len(samples)
        return len(samples)

    def append_samples(self, samples: np.ndarray) -> int:
        """Append float32 1D array of audio samples."""
        if samples.ndim != 1:
            samples = samples.flatten()
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._chunks.append(samples)
        self._total_samples += len(samples)
        return len(samples)

    def get_uncommitted_samples(self) -> np.ndarray:
        """Return array of uncommitted audio samples."""
        if not self._chunks or self._total_samples <= self._committed_samples:
            return np.empty(0, dtype=np.float32)

        # Concatenate and slice uncommitted region
        all_samples = np.concatenate(self._chunks)
        return all_samples[self._committed_samples:]

    def get_all_samples(self) -> np.ndarray:
        """Return entire accumulated audio session as a single 1D array."""
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def commit(self, count: int | None = None) -> None:
        """Mark uncommitted samples as committed."""
        if count is None:
            self._committed_samples = self._total_samples
        else:
            self._committed_samples = min(self._total_samples, self._committed_samples + count)

        # Compact chunks list if committed samples grow large (prevent memory unbounded growth)
        if self._committed_samples > self.sample_rate * 60 and len(self._chunks) > 20:
            all_samples = np.concatenate(self._chunks)
            # Retain audio from committed_samples onward
            remaining = all_samples[self._committed_samples:]
            self._chunks = [remaining]
            self._total_samples = len(remaining)
            self._committed_samples = 0

    @staticmethod
    def calculate_rms(samples: np.ndarray) -> float:
        """Compute root-mean-square (RMS) energy of audio samples."""
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    def clear(self) -> None:
        """Reset buffer state."""
        self._chunks.clear()
        self._total_samples = 0
        self._committed_samples = 0


class LiveTranscriptionSession:
    """Manages an active real-time streaming transcription session."""

    def __init__(
        self,
        session_id: str | None = None,
        whisper_engine: str = "faster-whisper",
        whisper_model: str = "base",
        language: str = "auto",
        vad_filter: bool = True,
        sample_rate: int = 16000,
        min_chunk_duration: float = 1.2,
        max_chunk_duration: float = 8.0,
        silence_commit_seconds: float = 0.8,
        silence_threshold_rms: float = 0.003,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.whisper_engine = whisper_engine
        self.whisper_model = whisper_model
        self.language = language
        self.vad_filter = vad_filter
        self.sample_rate = sample_rate
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.silence_commit_seconds = silence_commit_seconds
        self.silence_threshold_rms = silence_threshold_rms

        self.buffer = AudioRingBuffer(sample_rate=sample_rate)
        self.segments: list[Segment] = []
        self.current_partial_text: str = ""
        self.committed_time_offset: float = 0.0
        self.last_speech_time: float = 0.0
        self._consecutive_silence_count: int = 0
        self._next_segment_id: int = 0

    def add_chunk(self, pcm_bytes: bytes) -> int:
        """Ingest incoming binary PCM chunk."""
        return self.buffer.append_pcm16(pcm_bytes)

    def should_process(self) -> bool:
        """Determine whether enough audio has arrived to run an inference pass."""
        return self.buffer.uncommitted_duration_seconds >= self.min_chunk_duration

    def step(self) -> list[dict[str, Any]]:
        """Run an incremental inference step on uncommitted audio.

        Returns a list of WebSocket event dictionaries (e.g. 'partial' or 'segment').
        """
        uncommitted = self.buffer.get_uncommitted_samples()
        uncommitted_dur = len(uncommitted) / self.sample_rate

        if uncommitted_dur < self.min_chunk_duration:
            return []

        # Evaluate energy of the trailing recent audio to detect speech pauses
        recent_window_samples = min(len(uncommitted), int(self.sample_rate * self.silence_commit_seconds))
        trailing_audio = uncommitted[-recent_window_samples:]
        trailing_rms = self.buffer.calculate_rms(trailing_audio)
        overall_rms = self.buffer.calculate_rms(uncommitted)

        is_pure_silence = overall_rms < self.silence_threshold_rms
        trailing_is_silent = trailing_rms < self.silence_threshold_rms

        # Silence Pause Handling: if speech was previously detected and user paused (trailing audio is silent)
        if self.current_partial_text and trailing_is_silent:
            self._consecutive_silence_count += 1
            if self._consecutive_silence_count >= 1:
                # Commit the pending partial
                seg = Segment(
                    id=self._next_segment_id,
                    start=round(self.committed_time_offset, 2),
                    end=round(self.committed_time_offset + uncommitted_dur, 2),
                    text=self.current_partial_text,
                )
                self._next_segment_id += 1
                self.segments.append(seg)
                self.committed_time_offset += uncommitted_dur
                self.buffer.commit(len(uncommitted))
                committed_text = self.current_partial_text
                self.current_partial_text = ""
                self._consecutive_silence_count = 0
                return [{
                    "event": "segment",
                    "segment": seg.model_dump(),
                    "text": committed_text,
                }]

        if is_pure_silence:
            # Overall silence with no text: advance buffer without running heavy Whisper model
            if uncommitted_dur >= self.max_chunk_duration:
                self.committed_time_offset += uncommitted_dur
                self.buffer.commit(len(uncommitted))
            return []

        # Speech detected: run Whisper inference on uncommitted audio
        self._consecutive_silence_count = 0
        try:
            transcriber = transcriber_factory.get_transcriber(self.whisper_engine)
            res = transcriber.transcribe(
                audio=uncommitted,
                model_name=self.whisper_model,
                language=self.language,
                vad_filter=self.vad_filter,
                word_timestamps=True,
            )
        except Exception as e:
            logger.warning("Streaming inference step error: %s", e)
            return []

        recognized_text = res.text.strip()
        if not recognized_text:
            return []

        # Check if current utterance can be finalized into a committed segment
        ends_with_terminal_punctuation = recognized_text.endswith((".", "!", "?"))
        should_commit = (
            (ends_with_terminal_punctuation and uncommitted_dur >= 1.5)
            or uncommitted_dur >= self.max_chunk_duration
        )

        events: list[dict[str, Any]] = []

        if should_commit:
            # Build committed segments with aligned absolute timestamps
            for s in res.segments:
                adjusted_words = []
                for w in s.words:
                    adjusted_words.append(Word(
                        word=w.word,
                        start=round(self.committed_time_offset + w.start, 3),
                        end=round(self.committed_time_offset + w.end, 3),
                        probability=w.probability,
                        speaker=w.speaker,
                    ))

                seg = Segment(
                    id=self._next_segment_id,
                    start=round(self.committed_time_offset + s.start, 2),
                    end=round(self.committed_time_offset + s.end, 2),
                    text=s.text.strip(),
                    words=adjusted_words,
                    confidence=s.confidence,
                )
                self._next_segment_id += 1
                self.segments.append(seg)
                events.append({
                    "event": "segment",
                    "segment": seg.model_dump(),
                    "text": seg.text,
                })

            # Advance buffer and offset
            self.committed_time_offset += uncommitted_dur
            self.buffer.commit(len(uncommitted))
            self.current_partial_text = ""
        else:
            # Emit live interim hypothesis
            self.current_partial_text = recognized_text
            events.append({
                "event": "partial",
                "text": recognized_text,
                "start": round(self.committed_time_offset, 2),
                "end": round(self.committed_time_offset + uncommitted_dur, 2),
            })

        return events

    def finish(self) -> TranscriptionResult:
        """Finalize the session, transcribing any remaining uncommitted audio."""
        uncommitted = self.buffer.get_uncommitted_samples()
        uncommitted_dur = len(uncommitted) / self.sample_rate

        if uncommitted_dur > 0.5:
            rms = self.buffer.calculate_rms(uncommitted)
            if rms >= self.silence_threshold_rms:
                try:
                    transcriber = transcriber_factory.get_transcriber(self.whisper_engine)
                    res = transcriber.transcribe(
                        audio=uncommitted,
                        model_name=self.whisper_model,
                        language=self.language,
                        vad_filter=self.vad_filter,
                        word_timestamps=True,
                    )
                    for s in res.segments:
                        adjusted_words = [
                            Word(
                                word=w.word,
                                start=round(self.committed_time_offset + w.start, 3),
                                end=round(self.committed_time_offset + w.end, 3),
                                probability=w.probability,
                                speaker=w.speaker,
                            )
                            for w in s.words
                        ]
                        seg = Segment(
                            id=self._next_segment_id,
                            start=round(self.committed_time_offset + s.start, 2),
                            end=round(self.committed_time_offset + s.end, 2),
                            text=s.text.strip(),
                            words=adjusted_words,
                            confidence=s.confidence,
                        )
                        self._next_segment_id += 1
                        self.segments.append(seg)
                except Exception as e:
                    logger.warning("Error finalizing remaining streaming buffer: %s", e)
            elif self.current_partial_text:
                seg = Segment(
                    id=self._next_segment_id,
                    start=round(self.committed_time_offset, 2),
                    end=round(self.committed_time_offset + uncommitted_dur, 2),
                    text=self.current_partial_text,
                )
                self._next_segment_id += 1
                self.segments.append(seg)

        # Assemble complete text
        full_text = " ".join(s.text.strip() for s in self.segments if s.text.strip())
        total_duration = round(self.buffer.duration_seconds, 2)

        return TranscriptionResult(
            text=full_text,
            segments=self.segments,
            language=self.language,
            duration=total_duration,
        )
