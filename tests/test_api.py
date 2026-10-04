"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Audio Transcriber" in response.text


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ffmpeg"]["available"] is True
    assert len(data["whisper_engines"]) >= 1
    assert len(data["llm_providers"]) >= 1


def test_llm_models_endpoint():
    response = client.get("/api/llm/models?provider=ollama")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "ollama"
    assert "models" in data


def test_api_transcribe_upload():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "synth.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5", "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
        )

        with open(wav_path, "rb") as f:
            response = client.post(
                "/api/transcribe",
                files={"file": ("synth.wav", f, "audio/wav")},
                data={"whisper_engine": "faster-whisper", "whisper_model": "tiny", "language": "en"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "filename" in data
        assert "text" in data
        assert "srt" in data
        assert "vtt" in data
        assert "segments" in data
        assert data.get("audio_enhanced") is False


def test_api_transcribe_with_clean_audio():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "synth_clean.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5", "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
        )

        with open(wav_path, "rb") as f:
            response = client.post(
                "/api/transcribe",
                files={"file": ("synth_clean.wav", f, "audio/wav")},
                data={
                    "whisper_engine": "faster-whisper",
                    "whisper_model": "tiny",
                    "language": "en",
                    "clean_audio": "true",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data.get("audio_enhanced") is True
        assert "text" in data


def test_api_jobs_create_with_clean_audio():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "synth_job.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.5", "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            capture_output=True,
        )

        with open(wav_path, "rb") as f:
            response = client.post(
                "/api/jobs",
                files={"file": ("synth_job.wav", f, "audio/wav")},
                data={
                    "whisper_engine": "faster-whisper",
                    "whisper_model": "tiny",
                    "language": "en",
                    "clean_audio": "true",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

