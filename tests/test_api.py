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


def test_api_models_endpoints():
    # Test GET /api/models/status
    res = client.get("/api/models/status")
    assert res.status_code == 200
    data = res.json()
    assert "idle_timeout_seconds" in data
    assert "cached_models" in data

    # Test POST /api/models/unload-idle
    unload_res = client.post("/api/models/unload-idle")
    assert unload_res.status_code == 200
    unload_data = unload_res.json()
    assert unload_data["status"] == "ok"
    assert "unloaded" in unload_data
    assert "remaining" in unload_data


def test_api_batch_jobs_lifecycle():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        wav1 = Path(tmpdir) / "b1.wav"
        wav2 = Path(tmpdir) / "b2.wav"
        for p in (wav1, wav2):
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=800:duration=0.3", "-ar", "16000", "-ac", "1", str(p)],
                check=True,
                capture_output=True,
            )

        with open(wav1, "rb") as f1, open(wav2, "rb") as f2:
            res = client.post(
                "/api/jobs/batch",
                files=[
                    ("files", ("b1.wav", f1, "audio/wav")),
                    ("files", ("b2.wav", f2, "audio/wav")),
                ],
                data={"whisper_engine": "faster-whisper", "whisper_model": "tiny", "language": "en"},
            )

        assert res.status_code == 200
        data = res.json()
        assert "batch_id" in data
        batch_id = data["batch_id"]
        assert data["status"] == "queued"
        assert data["summary"]["total_jobs"] == 2

        # Test GET /api/batches
        list_res = client.get("/api/batches")
        assert list_res.status_code == 200
        batches = list_res.json().get("batches", [])
        assert any(b["batch_id"] == batch_id for b in batches)

        # Test GET /api/batches/{batch_id}
        detail_res = client.get(f"/api/batches/{batch_id}")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert detail["batch_id"] == batch_id
        assert detail["total_jobs"] == 2
        assert len(detail["jobs"]) == 2

    # Test 404 on nonexistent batch
    not_found_res = client.get("/api/batches/nonexistent_batch")
    assert not_found_res.status_code == 404


def test_api_batch_zip_export():
    import io
    import zipfile
    from jobs import job_manager

    # Set up a batch with completed mock result
    batch, jobs = job_manager.create_batch(["mock_sample.wav"])
    job = jobs[0]
    job.status = "completed"
    job.result = {
        "text": "Transcribed batch sample.",
        "srt": "1\n00:00:00,000 --> 00:00:01,000\nTranscribed batch sample.",
        "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTranscribed batch sample.",
    }

    # Test GET /api/batches/{batch_id}/export-zip
    export_res = client.get(f"/api/batches/{batch.batch_id}/export-zip")
    assert export_res.status_code == 200
    assert export_res.headers["content-type"] == "application/zip"
    assert f"batch_{batch.batch_id}_transcripts.zip" in export_res.headers.get("content-disposition", "")

    # Validate zip file contents
    with zipfile.ZipFile(io.BytesIO(export_res.content), "r") as zf:
        names = zf.namelist()
        assert "batch_manifest.json" in names
        assert any("mock_sample.txt" in n for n in names)

    # Test 404 for nonexistent batch zip export
    export_404 = client.get("/api/batches/nonexistent/export-zip")
    assert export_404.status_code == 404


def test_api_history_and_search():
    from jobs import job_manager

    # Create and persist a completed job with distinct keywords
    job = job_manager.create_job("strategy_q4.mp3")
    job.status = "completed"
    job.progress = 100
    job.duration = 84.5
    job.processing_time = 4.2
    job.result = {
        "text": "The key breakthrough was neural audio synthesis with low latency.",
        "summary": "Breakthrough in synthesis architecture.",
        "language": "en",
        "engine_used": "faster-whisper",
        "model_used": "base",
    }
    job_manager.store.save_job(job)

    # 1. Test GET /api/history
    res_hist = client.get("/api/history?page=1&limit=10")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert "items" in hist_data
    assert "total" in hist_data
    assert hist_data["total"] >= 1
    assert any(item["job_id"] == job.job_id for item in hist_data["items"])

    # 2. Test GET /api/history/stats
    res_stats = client.get("/api/history/stats")
    assert res_stats.status_code == 200
    stats_data = res_stats.json()
    assert stats_data["total_jobs"] >= 1

    # 3. Test GET /api/search (FTS5)
    res_search = client.get("/api/search?q=breakthrough")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total_matches"] >= 1
    matched = next((r for r in search_data["results"] if r["job_id"] == job.job_id), None)
    assert matched is not None
    assert "breakthrough" in matched["text_snippet"].lower() or "breakthrough" in matched["summary_snippet"].lower()

    # 4. Test DELETE /api/history/{job_id}
    del_res = client.delete(f"/api/history/{job.job_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_job_id"] == job.job_id

    # Verify 404 on subsequent delete
    del_404 = client.delete(f"/api/history/{job.job_id}")
    assert del_404.status_code == 404


def test_api_process_llm_persists_on_demand_output_and_indexes_fts():
    from unittest.mock import MagicMock, patch
    from jobs import job_manager

    job = job_manager.create_job("deep_learning_brief.wav")
    job.status = "completed"
    job.result = {
        "text": "Deep neural architectures enable unprecedented quantum speedups.",
        "language": "en",
    }
    job_manager.store.save_job(job)

    # Mock provider streaming
    mock_prov = MagicMock()
    async def mock_stream(*args, **kwargs):
        yield "Key "
        yield "Takeaway: "
        yield "Quantum "
        yield "advantage."

    mock_prov.generate_stream = mock_stream

    with patch("main.llm_registry.get_provider", return_value=mock_prov):
        res = client.post(
            "/api/process-llm",
            json={
                "job_id": job.job_id,
                "text": job.result["text"],
                "action": "summary",
                "detail_level": "bullets",
                "provider": "ollama",
            },
        )
        assert res.status_code == 200
        # Consume streaming response
        content = res.text
        assert "[DONE]" in content

    # Verify that the job was updated in persistent storage
    reloaded = job_manager.store.get_job(job.job_id)
    assert reloaded is not None
    assert reloaded.result is not None
    assert reloaded.result["summary"] == "Key Takeaway: Quantum advantage."
    assert reloaded.result["summaries"]["bullets"] == "Key Takeaway: Quantum advantage."

    # Verify searchable in FTS5
    search_res = client.get("/api/search?q=advantage")
    assert search_res.status_code == 200
    search_json = search_res.json()
    assert search_json["total_matches"] >= 1
    assert any(r["job_id"] == job.job_id for r in search_json["results"])





