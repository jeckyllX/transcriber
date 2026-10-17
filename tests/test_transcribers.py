"""Tests for Transcriber outputs, timestamps, and formatting."""

import json
from transcribers.base import Segment, TranscriptionResult
from transcribers.factory import TranscriberFactory


def test_transcription_result_formatters():
    from transcribers.base import Word

    words_seg1 = [
        Word(word="Hello", start=1.234, end=2.000, probability=0.95),
        Word(word="world!", start=2.100, end=4.567, probability=0.98),
    ]
    segments = [
        Segment(id=1, start=1.234, end=4.567, text="Hello world!", words=words_seg1),
        Segment(id=2, start=5.000, end=7.890, text="This is a test transcription."),
    ]
    res = TranscriptionResult(
        text="Hello world! This is a test transcription.",
        segments=segments,
        language="en",
        duration=7.89,
    )

    # Plain text
    assert res.to_txt() == "Hello world! This is a test transcription."

    # SRT formatting
    srt = res.to_srt()
    assert "00:00:01,234 --> 00:00:04,567" in srt
    assert "Hello world!" in srt
    assert "00:00:05,000 --> 00:00:07,890" in srt

    # VTT formatting
    vtt = res.to_vtt()
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.234 --> 00:00:04.567" in vtt

    # Word-level WebVTT formatting
    word_vtt = res.to_word_vtt()
    assert word_vtt.startswith("WEBVTT")
    assert "<00:00:01.234>Hello" in word_vtt
    assert "<00:00:02.100>world!" in word_vtt

    # ASS karaoke formatting
    ass = res.to_ass("Test Meeting")
    assert "[Script Info]" in ass
    assert "Title: Test Meeting" in ass
    assert "[V4+ Styles]" in ass
    assert "Dialogue: 0,0:00:01.23,0:00:04.57,Karaoke,,0,0,0,,{\\k77}Hello {\\k247}world!" in ass
    assert "Dialogue: 0,0:00:05.00,0:00:07.89,Default,,0,0,0,,This is a test transcription." in ass

    # JSON formatting
    data = json.loads(res.to_json())
    assert data["language"] == "en"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hello world!"
    assert len(data["segments"][0]["words"]) == 2
    assert data["segments"][0]["words"][0]["word"] == "Hello"
    assert data["segments"][0]["words"][0]["probability"] == 0.95


def test_transcriber_factory():
    factory = TranscriberFactory()
    engines = factory.list_engines()
    engine_ids = [e["id"] for e in engines]
    assert "whisper.cpp" in engine_ids
    assert "faster-whisper" in engine_ids

    # Should safely retrieve the available engine (faster-whisper in this env)
    active = factory.get_transcriber()
    assert active.is_available()


def test_managed_model_pool_lease_and_release():
    from transcribers.model_pool import ManagedModelPool
    import time

    loaded = []
    disposed = []

    def loader(name):
        loaded.append(name)
        return {"name": name, "weights": [1, 2, 3]}

    def disposer(inst):
        disposed.append(inst["name"])

    pool = ManagedModelPool(loader=loader, disposer=disposer, name="test-pool")

    # Initial state
    assert not pool.is_in_use("tiny")
    assert pool.get_status() == {}

    # Lease model
    with pool.lease("tiny") as model:
        assert model["name"] == "tiny"
        assert pool.is_in_use("tiny")
        status = pool.get_status()
        assert status["tiny"]["in_use"] is True
        assert status["tiny"]["active_leases"] == 1

    # After lease exit
    assert not pool.is_in_use("tiny")
    status = pool.get_status()
    assert status["tiny"]["in_use"] is False
    assert status["tiny"]["active_leases"] == 0
    assert len(loaded) == 1
    assert len(disposed) == 0


def test_managed_model_pool_eviction_protection_while_processing():
    from transcribers.model_pool import ManagedModelPool
    import time

    disposed = []
    pool = ManagedModelPool(
        loader=lambda name: {"name": name},
        disposer=lambda inst: disposed.append(inst["name"]),
        name="test-pool",
    )

    # Start an active lease
    with pool.lease("base") as model:
        assert pool.is_in_use("base")

        # Attempt to evict idle models with a 0-second threshold while leased
        evicted = pool.evict_idle(timeout_seconds=0.001)
        assert "base" not in evicted
        assert pool.is_in_use("base")
        assert len(disposed) == 0

        # Attempt manual unload while leased
        unloaded = pool.unload_idle()
        assert "base" not in unloaded
        assert len(disposed) == 0

    # Model is now released (idle). Sleep briefly past threshold
    time.sleep(0.05)
    evicted = pool.evict_idle(timeout_seconds=0.02)
    assert "base" in evicted
    assert "base" in disposed
    assert pool.get_status() == {}


def test_managed_model_pool_exception_safety():
    from transcribers.model_pool import ManagedModelPool
    import pytest

    pool = ManagedModelPool(loader=lambda name: {"name": name}, name="test-pool")

    with pytest.raises(ValueError, match="simulated failure"):
        with pool.lease("small"):
            assert pool.is_in_use("small")
            raise ValueError("simulated failure")

    # Ensure lease count decremented back to 0 despite exception
    assert not pool.is_in_use("small")
    status = pool.get_status()
    assert status["small"]["active_leases"] == 0


def test_faster_whisper_and_factory_eviction_lifecycle():
    from transcribers.faster_whisper import FasterWhisperTranscriber
    from transcribers.factory import TranscriberFactory
    import time

    transcriber = FasterWhisperTranscriber()
    # Mock pool loader for test
    transcriber.pool.loader = lambda name: {"mock_model": name}

    with transcriber.pool.lease("tiny") as model:
        assert model["mock_model"] == "tiny"
        # While leased, cannot be evicted
        assert transcriber.evict_idle_models(timeout_seconds=0.001) == []
        assert transcriber.unload_all_idle() == []
        assert transcriber.unload_model("tiny") is False

    # Once released, can be unloaded
    time.sleep(0.02)
    evicted = transcriber.evict_idle_models(timeout_seconds=0.01)
    assert "tiny" in evicted

    # Test factory delegation
    factory = TranscriberFactory()
    status = factory.get_cached_models_status()
    assert isinstance(status, dict)
    unloaded = factory.unload_all_idle()
    assert isinstance(unloaded, dict)


def test_timestamp_boundary_rounding_rollover():
    """Verify that fractional second rounding near 0.9995+ rolls over seconds and minutes cleanly."""
    from transcribers.base import TranscriptionResult

    # 1.9996s -> should roll over to 2.000s, NOT 00:00:01,1000
    assert TranscriptionResult._format_timestamp(1.9996, srt_format=True) == "00:00:02,000"
    assert TranscriptionResult._format_timestamp(1.9996, srt_format=False) == "00:00:02.000"

    # 59.9998s -> should roll over to 1 minute, 00:01:00,000
    assert TranscriptionResult._format_timestamp(59.9998, srt_format=True) == "00:01:00,000"

    # 3599.9999s -> should roll over to 1 hour, 01:00:00,000
    assert TranscriptionResult._format_timestamp(3599.9999, srt_format=True) == "01:00:00,000"

    # ASS centisecond rollover: 59.996s -> 0:01:00.00
    assert TranscriptionResult._format_ass_timestamp(59.996) == "0:01:00.00"
    # Zero second edge case
    assert TranscriptionResult._format_timestamp(0.0) == "00:00:00,000"
    assert TranscriptionResult._format_ass_timestamp(0.0) == "0:00:00.00"


