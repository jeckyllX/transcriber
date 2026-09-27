"""Temporal alignment between speaker diarization intervals and Whisper segments/words."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transcribers.base import Segment
from diarization.base import DiarizationResult, SpeakerInterval


def compute_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate temporal intersection in seconds between two intervals."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def align_speakers_to_segments(
    segments: list[Segment],
    diarization: DiarizationResult,
    default_speaker: str = "Speaker 0",
) -> list[Segment]:
    """Assign speaker labels to Whisper Segment and Word models based on temporal overlap.

    Args:
        segments: List of transcribed segments with start, end, and optional word timestamps.
        diarization: DiarizationResult containing speaker time intervals.
        default_speaker: Fallback speaker label when no intervals overlap.

    Returns:
        The mutated or updated segments list with speaker fields populated.
    """
    if not segments:
        return segments

    intervals = diarization.intervals
    if not intervals:
        # No speaker intervals detected; assign default speaker
        for seg in segments:
            seg.speaker = default_speaker
            for w in seg.words:
                w.speaker = default_speaker
        return segments

    prev_speaker = default_speaker

    for seg in segments:
        # Step 1: Align individual words if word timestamps exist
        speaker_duration: dict[str, float] = {}

        for w in seg.words:
            best_word_speaker: str | None = None
            best_word_overlap = 0.0

            for interval in intervals:
                overlap = compute_overlap(w.start, w.end, interval.start, interval.end)
                if overlap > best_word_overlap:
                    best_word_overlap = overlap
                    best_word_speaker = interval.speaker

            if best_word_speaker and best_word_overlap > 0:
                w.speaker = best_word_speaker
                w_dur = max(0.05, w.end - w.start)
                speaker_duration[best_word_speaker] = speaker_duration.get(best_word_speaker, 0.0) + w_dur

        # Step 2: Determine segment speaker from words majority or segment-level overlap
        assigned_speaker: str | None = None

        if speaker_duration:
            # Pick speaker with the most spoken duration in this segment
            assigned_speaker = max(speaker_duration.items(), key=lambda kv: kv[1])[0]
        else:
            # Fallback to segment-level overlap
            best_seg_speaker: str | None = None
            best_seg_overlap = 0.0

            for interval in intervals:
                overlap = compute_overlap(seg.start, seg.end, interval.start, interval.end)
                if overlap > best_seg_overlap:
                    best_seg_overlap = overlap
                    best_seg_speaker = interval.speaker

            if best_seg_speaker and best_seg_overlap > 0:
                assigned_speaker = best_seg_speaker
            else:
                # Find closest interval in time
                min_dist = float("inf")
                closest_speaker: str | None = None
                for interval in intervals:
                    dist = max(0.0, interval.start - seg.end, seg.start - interval.end)
                    if dist < min_dist:
                        min_dist = dist
                        closest_speaker = interval.speaker

                if closest_speaker and min_dist <= 3.0:
                    assigned_speaker = closest_speaker
                else:
                    assigned_speaker = prev_speaker

        seg.speaker = assigned_speaker
        prev_speaker = assigned_speaker

        # Step 3: Backfill any words in the segment that lacked direct interval overlap
        for w in seg.words:
            if not getattr(w, "speaker", None):
                w.speaker = assigned_speaker

    return segments
