"""Unit and integration tests for speaker diarization and speaker-aware features."""

import pytest
import numpy as np
from diarization.base import SpeakerInterval, DiarizationResult
from diarization.factory import diarizer_factory
from diarization.alignment import compute_overlap, align_speakers_to_segments
from diarization.sherpa_diarizer import SherpaDiarizer
from transcribers.base import Segment, Word, TranscriptionResult
from jobs import job_manager


def test_diarization_data_models():
    interval1 = SpeakerInterval(start=0.0, end=2.5, speaker="Speaker 0", confidence=0.92)
    interval2 = SpeakerInterval(start=2.8, end=5.0, speaker="Speaker 1", confidence=0.88)
    interval3 = SpeakerInterval(start=5.2, end=7.0, speaker="Speaker 0", confidence=0.95)

    result = DiarizationResult(
        num_speakers=2,
        intervals=[interval1, interval2, interval3]
    )

    assert result.num_speakers == 2
    assert result.speaker_names == ["Speaker 0", "Speaker 1"]
    assert len(result.intervals) == 3


def test_compute_overlap():
    assert compute_overlap(0.0, 5.0, 2.0, 4.0) == 2.0
    assert compute_overlap(0.0, 2.0, 3.0, 5.0) == 0.0
    assert compute_overlap(1.0, 4.0, 2.0, 6.0) == 2.0
    assert compute_overlap(5.0, 10.0, 4.0, 6.0) == 1.0


def test_align_speakers_to_segments_without_words():
    segments = [
        Segment(id=0, start=0.0, end=3.0, text="Hello there."),
        Segment(id=1, start=3.5, end=6.0, text="General Kenobi!"),
    ]
    diarization = DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=2.8, speaker="Speaker 0"),
            SpeakerInterval(start=3.2, end=6.2, speaker="Speaker 1"),
        ]
    )

    aligned = align_speakers_to_segments(segments, diarization)
    assert aligned[0].speaker == "Speaker 0"
    assert aligned[1].speaker == "Speaker 1"


def test_align_speakers_to_segments_with_words():
    words1 = [
        Word(word="Good", start=0.0, end=0.5),
        Word(word="morning", start=0.6, end=1.2),
        Word(word="everyone", start=1.3, end=2.0),
    ]
    words2 = [
        Word(word="Welcome", start=2.5, end=3.0),
        Word(word="back", start=3.1, end=3.8),
    ]
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Good morning everyone", words=words1),
        Segment(id=1, start=2.5, end=3.8, text="Welcome back", words=words2),
    ]
    diarization = DiarizationResult(
        num_speakers=2,
        intervals=[
            SpeakerInterval(start=0.0, end=2.2, speaker="Speaker 0"),
            SpeakerInterval(start=2.4, end=4.0, speaker="Speaker 1"),
        ]
    )

    aligned = align_speakers_to_segments(segments, diarization)
    assert aligned[0].speaker == "Speaker 0"
    assert aligned[0].words[0].speaker == "Speaker 0"
    assert aligned[1].speaker == "Speaker 1"
    assert aligned[1].words[0].speaker == "Speaker 1"


def test_align_speakers_empty_intervals_fallback():
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Single speaker recording."),
    ]
    diarization = DiarizationResult(num_speakers=0, intervals=[])

    aligned = align_speakers_to_segments(segments, diarization, default_speaker="Speaker 0")
    assert aligned[0].speaker == "Speaker 0"


def test_transcription_result_speaker_formatting():
    seg1 = Segment(id=0, start=0.0, end=2.0, text="First utterance", speaker="Speaker 0")
    seg2 = Segment(id=1, start=2.2, end=4.0, text="Second utterance", speaker="Speaker 0")
    seg3 = Segment(id=2, start=4.5, end=7.0, text="Reply from second speaker", speaker="Speaker 1")

    res = TranscriptionResult(
        text="First utterance Second utterance Reply from second speaker",
        segments=[seg1, seg2, seg3],
        duration=7.0,
    )

    # Test speaker dialogue turns
    turns = res.get_speaker_turns()
    assert len(turns) == 2
    assert turns[0]["speaker"] == "Speaker 0"
    assert turns[0]["text"] == "First utterance Second utterance"
    assert turns[0]["start"] == 0.0
    assert turns[0]["end"] == 4.0
    assert turns[1]["speaker"] == "Speaker 1"
    assert turns[1]["text"] == "Reply from second speaker"

    # Test TXT formatting
    txt = res.to_txt()
    assert "[Speaker 0]:" in txt
    assert "[Speaker 1]:" in txt

    # Test SRT formatting
    srt = res.to_srt()
    assert "Speaker 0: First utterance" in srt
    assert "Speaker 1: Reply from second speaker" in srt

    # Test WebVTT formatting
    vtt = res.to_vtt()
    assert "<v Speaker 0>First utterance</v>" in vtt
    assert "<v Speaker 1>Reply from second speaker</v>" in vtt

    # Test ASS formatting
    ass = res.to_ass()
    assert "Karaoke,Speaker 0" in ass or "Default,Speaker 0" in ass
    assert "Karaoke,Speaker 1" in ass or "Default,Speaker 1" in ass


def test_diarizer_factory():
    engines = diarizer_factory.list_engines()
    assert any(e["name"] == "sherpa-onnx" for e in engines)

    diarizer = diarizer_factory.get_diarizer("sherpa-onnx")
    assert isinstance(diarizer, SherpaDiarizer)
    assert diarizer.is_available() is True


def test_sherpa_diarizer_inference_dummy_silence():
    diarizer = diarizer_factory.get_diarizer("sherpa-onnx")
    # 2 seconds of silence (16kHz float32)
    silence = np.zeros(32000, dtype=np.float32)
    res = diarizer.diarize(silence)
    assert isinstance(res, DiarizationResult)
    assert isinstance(res.intervals, list)


def test_job_manager_rename_speaker():
    job = job_manager.create_job("meeting.wav")
    job.result = {
        "text": "[Speaker 0]: Hello\n\n[Speaker 1]: World",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello", "speaker": "Speaker 0", "words": [{"word": "Hello", "start": 0.0, "end": 1.0, "speaker": "Speaker 0"}]},
            {"id": 1, "start": 1.5, "end": 2.5, "text": "World", "speaker": "Speaker 1", "words": [{"word": "World", "start": 1.5, "end": 2.5, "speaker": "Speaker 1"}]},
        ],
        "language": "en",
        "duration": 2.5,
    }

    updated_result = job_manager.rename_speaker(job.job_id, "Speaker 0", "Alice")
    assert updated_result is not None
    assert updated_result["segments"][0]["speaker"] == "Alice"
    assert updated_result["segments"][0]["words"][0]["speaker"] == "Alice"
    assert updated_result["segments"][1]["speaker"] == "Speaker 1"
    assert "Alice: Hello" in updated_result["srt"]
    assert "<v Alice>Hello</v>" in updated_result["vtt"]
    assert "[Alice]:" in updated_result["text"]
    assert updated_result["speaker_turns"][0]["speaker"] == "Alice"
