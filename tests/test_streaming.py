"""Automated unit and integration tests for real-time WebSocket streaming."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from transcribers.base import Segment, TranscriptionResult, Word
from transcribers.streaming import AudioRingBuffer, LiveTranscriptionSession


def test_audio_ring_buffer_append_pcm16():
    buffer = AudioRingBuffer(sample_rate=16000)
    assert buffer.duration_seconds == 0.0
    assert buffer.uncommitted_samples_count == 0

    # 1 second of 16kHz 16-bit PCM (16000 samples * 2 bytes = 32000 bytes)
    # Generate a 440Hz sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    pcm_bytes = sine.tobytes()

    samples_added = buffer.append_pcm16(pcm_bytes)
    assert samples_added == 16000
    assert buffer.total_samples == 16000
    assert buffer.duration_seconds == 1.0
    assert buffer.uncommitted_duration_seconds == 1.0

    # Verify RMS calculation
    uncommitted = buffer.get_uncommitted_samples()
    rms = buffer.calculate_rms(uncommitted)
    assert rms > 0.1  # Significant energy for sine wave

    # Test commit
    buffer.commit(8000)
    assert buffer.committed_samples == 8000
    assert buffer.uncommitted_samples_count == 8000
    assert buffer.uncommitted_duration_seconds == 0.5


def test_audio_ring_buffer_silence_rms():
    buffer = AudioRingBuffer(sample_rate=16000)
    # Pure zero silence
    silence = np.zeros(8000, dtype=np.int16).tobytes()
    buffer.append_pcm16(silence)
    uncommitted = buffer.get_uncommitted_samples()
    rms = buffer.calculate_rms(uncommitted)
    assert rms == 0.0


def test_live_transcription_session_lifecycle():
    session = LiveTranscriptionSession(
        session_id="test-session-1",
        whisper_engine="faster-whisper",
        whisper_model="base",
        min_chunk_duration=1.0,
    )

    assert session.session_id == "test-session-1"
    assert not session.should_process()

    # Append 0.5s audio (not enough to process yet)
    pcm_0_5s = np.zeros(8000, dtype=np.int16).tobytes()
    session.add_chunk(pcm_0_5s)
    assert not session.should_process()
    assert session.step() == []

    # Append another 1.0s audio (total 1.5s >= 1.0s min_chunk_duration)
    pcm_1_0s = np.zeros(16000, dtype=np.int16).tobytes()
    session.add_chunk(pcm_1_0s)
    assert session.should_process()


def test_live_transcription_session_mock_inference():
    session = LiveTranscriptionSession(
        session_id="mock-session",
        min_chunk_duration=1.0,
        silence_threshold_rms=0.01,
    )

    # 1.5s of sine wave audio (speech-like energy)
    t = np.linspace(0, 1.5, 24000, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    session.add_chunk(sine.tobytes())

    mock_result = TranscriptionResult(
        text="Hello world.",
        segments=[
            Segment(
                id=0,
                start=0.1,
                end=1.2,
                text="Hello world.",
                words=[
                    Word(word="Hello", start=0.1, end=0.6, probability=0.98),
                    Word(word="world.", start=0.7, end=1.2, probability=0.99),
                ],
            )
        ],
        language="en",
        duration=1.5,
    )

    with patch("transcribers.streaming.transcriber_factory.get_transcriber") as mock_factory:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_result
        mock_factory.return_value = mock_transcriber

        events = session.step()
        assert len(events) == 1
        assert events[0]["event"] == "segment"
        assert events[0]["text"] == "Hello world."
        assert len(session.segments) == 1
        assert session.segments[0].text == "Hello world."

    # Finalize
    res = session.finish()
    assert res.text == "Hello world."
    assert len(res.segments) == 1


def test_live_transcription_session_partial_and_silence_commit():
    session = LiveTranscriptionSession(
        session_id="partial-session",
        min_chunk_duration=1.0,
        silence_threshold_rms=0.01,
    )

    # 1.1s of audio with text without punctuation
    t = np.linspace(0, 1.1, 17600, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)
    session.add_chunk(sine.tobytes())

    mock_partial = TranscriptionResult(
        text="speaking right now without pause",
        segments=[
            Segment(id=0, start=0.0, end=1.1, text="speaking right now without pause")
        ],
        language="en",
        duration=1.1,
    )

    with patch("transcribers.streaming.transcriber_factory.get_transcriber") as mock_factory:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = mock_partial
        mock_factory.return_value = mock_transcriber

        events = session.step()
        assert len(events) == 1
        assert events[0]["event"] == "partial"
        assert events[0]["text"] == "speaking right now without pause"
        assert session.current_partial_text == "speaking right now without pause"

    # Now feed silence:
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    session.add_chunk(silence)
    # Silence pause detected -> commits pending partial immediately!
    ev1 = session.step()
    assert len(ev1) == 1
    assert ev1[0]["event"] == "segment"
    assert ev1[0]["text"] == "speaking right now without pause"
    assert session.current_partial_text == ""


def test_websocket_transcribe_endpoint():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    with client.websocket_connect("/api/ws/transcribe") as websocket:
        # Handshake: server sends ready event
        ready_msg = websocket.receive_json()
        assert ready_msg["event"] == "ready"
        assert ready_msg["sample_rate"] == 16000
        assert "session_id" in ready_msg

        # Configure session
        websocket.send_json({
            "action": "start",
            "whisper_engine": "faster-whisper",
            "whisper_model": "base",
            "language": "en"
        })
        config_msg = websocket.receive_json()
        assert config_msg["event"] == "configured"
        assert config_msg["model"] == "base"
        assert config_msg["language"] == "en"

        # Send audio PCM bytes (1 second silence)
        pcm_bytes = np.zeros(16000, dtype=np.int16).tobytes()
        websocket.send_bytes(pcm_bytes)

        # Send stop action
        websocket.send_json({"action": "stop"})
        completed_msg = websocket.receive_json()
        assert completed_msg["event"] == "completed"
        assert "result" in completed_msg
        assert "duration" in completed_msg["result"]


