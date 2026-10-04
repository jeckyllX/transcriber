"""Tests for AudioProcessor media probing and conversion."""

import subprocess
import tempfile
from pathlib import Path
import pytest
from audio_processor import AudioProcessor


@pytest.fixture
def synthetic_audio_files(tmp_path: Path):
    """Generate synthetic sine-wave audio files using ffmpeg for testing."""
    mp3_path = tmp_path / "test_sine.mp3"
    wav_path = tmp_path / "test_stereo.wav"

    # Generate 1.5s stereo 44.1kHz MP3
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=1000:duration=1.5",
            "-ac", "2", "-ar", "44100",
            str(mp3_path),
        ],
        capture_output=True,
        check=True,
    )

    # Generate 1.0s stereo 48kHz WAV
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1.0",
            "-ac", "2", "-ar", "48000",
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )

    return {"mp3": mp3_path, "wav": wav_path}


def test_probe_media(synthetic_audio_files):
    processor = AudioProcessor()
    meta = processor.probe_media(synthetic_audio_files["mp3"])
    assert meta.duration > 1.0
    assert meta.channels == 2
    assert meta.sample_rate == 44100
    assert meta.size_bytes > 0


def test_convert_to_whisper_wav(synthetic_audio_files, tmp_path: Path):
    processor = AudioProcessor()
    dest = tmp_path / "converted_whisper.wav"

    converted = processor.convert_to_whisper_wav(synthetic_audio_files["mp3"], output_path=dest)
    assert converted.exists()
    assert converted.stat().st_size > 0

    # Probe the converted output to verify 16kHz Mono specification
    converted_meta = processor.probe_media(converted)
    assert converted_meta.sample_rate == 16000
    assert converted_meta.channels == 1
    assert converted_meta.duration > 1.0


def test_convert_to_pcm_array(synthetic_audio_files):
    processor = AudioProcessor()
    pcm_array, meta = processor.convert_to_pcm_array(synthetic_audio_files["mp3"])
    assert pcm_array.ndim == 1
    assert pcm_array.dtype.name == "float32"
    assert len(pcm_array) > 0
    assert meta.duration > 1.0


def test_convert_to_whisper_wav_with_enhancement(synthetic_audio_files, tmp_path: Path):
    processor = AudioProcessor()
    dest = tmp_path / "enhanced_whisper.wav"

    converted = processor.convert_to_whisper_wav(
        synthetic_audio_files["mp3"],
        output_path=dest,
        enhance_audio=True,
    )
    assert converted.exists()
    assert converted.stat().st_size > 0

    converted_meta = processor.probe_media(converted)
    assert converted_meta.sample_rate == 16000
    assert converted_meta.channels == 1
    assert converted_meta.duration > 1.0


def test_convert_to_pcm_array_with_enhancement(synthetic_audio_files):
    processor = AudioProcessor()
    pcm_array, meta = processor.convert_to_pcm_array(
        synthetic_audio_files["mp3"],
        enhance_audio=True,
    )
    assert pcm_array.ndim == 1
    assert pcm_array.dtype.name == "float32"
    assert len(pcm_array) > 0
    assert meta.duration > 1.0


def test_enhancement_graceful_fallback_on_invalid_filter(synthetic_audio_files, tmp_path: Path):
    # An intentionally invalid filter syntax that FFmpeg will reject
    processor = AudioProcessor(enhancement_filter="nonexistent_filter_xyz=999")
    dest = tmp_path / "fallback.wav"

    # Should not raise AudioProcessorError; must gracefully fall back to unfiltered conversion
    converted = processor.convert_to_whisper_wav(
        synthetic_audio_files["wav"],
        output_path=dest,
        enhance_audio=True,
    )
    assert converted.exists()
    assert converted.stat().st_size > 0

    # Also verify streaming PCM array fallback
    pcm_array, meta = processor.convert_to_pcm_array(
        synthetic_audio_files["wav"],
        enhance_audio=True,
    )
    assert len(pcm_array) > 0
    assert meta.duration > 0.5

