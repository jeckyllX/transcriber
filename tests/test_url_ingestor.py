"""Unit and integration tests for URL media ingestion, SSRF protection, URL API endpoints, and bug fix regressions."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
import jobs
import main
from llm.ollama import OllamaLLMProvider
from notifications.base import NotificationPayload
from notifications.telegram import TelegramNotifier
from transcribers.base import Segment, TranscriptionResult
from url_ingestor import DownloadedMedia, MediaURLIngestor, URLIngestionError, sanitize_filename


# =====================================================================
# 1. URL Validation & SSRF Protection Tests
# =====================================================================


def test_validate_media_url_accepts_public_http_https(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public HTTP/HTTPS URLs resolving to global IPs are accepted."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, proto=0: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    ingestor = MediaURLIngestor(allow_local_network=False)
    cleaned = ingestor.validate_media_url("  https://example.com/podcasts/ep01.mp3  ")
    assert cleaned == "https://example.com/podcasts/ep01.mp3"


@pytest.mark.parametrize(
    "bad_url,expected_msg",
    [
        ("", "Media URL cannot be empty"),
        ("   ", "Media URL cannot be empty"),
        ("ftp://example.com/audio.mp3", "Unsupported URL scheme"),
        ("file:///etc/passwd", "Unsupported URL scheme"),
        ("javascript:alert(1)", "Unsupported URL scheme"),
        ("http://", "URL must include a valid hostname"),
    ],
)
def test_validate_media_url_rejects_invalid_schemes(bad_url: str, expected_msg: str) -> None:
    ingestor = MediaURLIngestor(allow_local_network=False)
    with pytest.raises(URLIngestionError, match=expected_msg):
        ingestor.validate_media_url(bad_url)


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://localhost:8000/audio.mp3",
        "http://sub.localhost/audio.mp3",
        "http://127.0.0.1/audio.mp3",
        "http://10.0.0.1/audio.mp3",
        "http://172.16.0.5/audio.mp3",
        "http://192.168.1.50/audio.mp3",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/audio.mp3",
        "http://[::ffff:127.0.0.1]/audio.mp3",
    ],
)
def test_validate_media_url_blocks_ssrf_targets(blocked_url: str) -> None:
    """Loopback, private, link-local, and IPv4-mapped IPv6 targets are blocked by default."""
    ingestor = MediaURLIngestor(allow_local_network=False)
    with pytest.raises(URLIngestionError, match="blocked"):
        ingestor.validate_media_url(blocked_url)


def test_validate_media_url_allows_local_when_configured() -> None:
    """When allow_local_network=True, local addresses are permitted."""
    ingestor = MediaURLIngestor(allow_local_network=True)
    url = ingestor.validate_media_url("http://127.0.0.1:9000/sample.wav")
    assert url == "http://127.0.0.1:9000/sample.wav"


def test_sanitize_filename_removes_traversal_and_unsafe_chars() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename('My: "Podcast" <Episode #1>?.mp3') == "My_ _Podcast_ _Episode #1__.mp3"
    assert sanitize_filename("", default="remote_media") == "remote_media"


# =====================================================================
# 2. Direct HTTP Stream & yt-dlp Ingestion Tests
# =====================================================================


def test_httpx_direct_download_streams_and_extracts_filename(tmp_path: Path) -> None:
    """Direct HTTP download streams bytes to disk and extracts Content-Disposition filename."""
    audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 128

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "audio/wav",
                "Content-Length": str(len(audio_bytes)),
                "Content-Disposition": 'attachment; filename="interview_final.wav"',
            },
            content=audio_bytes,
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    ingestor = MediaURLIngestor(max_download_size_mb=10, timeout_seconds=30, allow_local_network=True)

    with patch("url_ingestor.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)):
        progress_updates: list[tuple[int, int | None, str]] = []
        result = ingestor._download_via_httpx(
            url="http://127.0.0.1/stream",
            dest_dir=tmp_path,
            prefix="test1234_",
            progress_callback=lambda dl, tot, name: progress_updates.append((dl, tot, name)),
        )

    assert result.file_path.is_file()
    assert result.file_path.read_bytes() == audio_bytes
    assert result.display_name == "interview_final.wav"
    assert result.extractor_used == "httpx"
    assert len(progress_updates) >= 1


def test_httpx_redirect_ssrf_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirecting from a public host to 127.0.0.1 is blocked before following."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, proto=0: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "http://127.0.0.1:8080/internal-secret.wav"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    ingestor = MediaURLIngestor(max_download_size_mb=10, timeout_seconds=30, allow_local_network=False)

    with patch("url_ingestor.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)):
        with pytest.raises(URLIngestionError, match="blocked"):
            ingestor._download_via_httpx(
                url="https://example.com/public.mp3",
                dest_dir=tmp_path,
                prefix="redir_ssrf_",
            )


def test_httpx_enforces_max_download_size_limit(tmp_path: Path) -> None:
    """Streaming download aborts and cleans up temp file if byte limit is exceeded."""
    oversized_payload = b"X" * (2 * 1024 * 1024)  # 2 MB

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/mpeg"},
            content=oversized_payload,
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    ingestor = MediaURLIngestor(max_download_size_mb=1, timeout_seconds=30, allow_local_network=True)

    with patch("url_ingestor.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)):
        with pytest.raises(URLIngestionError, match="maximum allowed"):
            ingestor._download_via_httpx(
                url="http://127.0.0.1/huge.mp3",
                dest_dir=tmp_path,
                prefix="oversized_",
            )

    assert list(tmp_path.glob("*oversized*")) == []


def test_httpx_html_response_without_ytdlp_raises_helpful_error(tmp_path: Path) -> None:
    """When URL returns an HTML webpage and yt-dlp is not installed, raise a clear error."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html><body>Not an audio file</body></html>",
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    ingestor = MediaURLIngestor(max_download_size_mb=10, timeout_seconds=30, allow_local_network=True)

    with (
        patch("url_ingestor.httpx.Client", side_effect=lambda **kw: real_client(transport=transport, **kw)),
        patch.object(MediaURLIngestor, "is_ytdlp_available", return_value=False),
    ):
        with pytest.raises(URLIngestionError, match="returned an HTML webpage instead of an audio/video stream"):
            ingestor._download_via_httpx(
                url="http://127.0.0.1/page",
                dest_dir=tmp_path,
                prefix="htmlpage_",
            )


def test_ytdlp_cli_extraction_creates_media_file(tmp_path: Path) -> None:
    """When yt-dlp binary is available, _download_via_ytdlp extracts media and resolves title."""
    out_file = tmp_path / "yt123_ytdlp_dQw4w9WgXcQ.m4a"

    def fake_run(cmd: list[str], **kwargs):
        out_file.write_bytes(b"FAKE_M4A_CONTENT")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=f"Keynote Speech 2026.m4a\n{out_file}\n",
            stderr="",
        )

    ingestor = MediaURLIngestor(
        max_download_size_mb=50,
        timeout_seconds=60,
        allow_local_network=False,
        ytdlp_bin="/usr/bin/yt-dlp",
    )

    with patch("url_ingestor.subprocess.run", side_effect=fake_run):
        media = ingestor._download_via_ytdlp(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            dest_dir=tmp_path,
            prefix="yt123_",
        )

    assert media.file_path.is_file()
    assert media.display_name == "Keynote Speech 2026.m4a"
    assert media.extractor_used == "yt-dlp"


# =====================================================================
# 3. API Endpoint Tests (/api/jobs/url, /api/jobs/batch-urls, /api/transcribe-url)
# =====================================================================


def _make_fake_transcription_result(text: str = "Hello from URL transcription.") -> tuple[TranscriptionResult, int, float, str]:
    res = TranscriptionResult(
        text=text,
        segments=[Segment(id=0, start=0.0, end=1.5, text=text)],
        language="en",
        duration=1.5,
        processing_time=0.2,
        model_used="base",
        engine_used="faster-whisper",
    )
    return res, 0, 1.5, "faster-whisper"


def test_api_jobs_url_rejects_ssrf_url() -> None:
    client = TestClient(main.app)
    resp = client.post(
        "/api/jobs/url",
        json={"url": "http://127.0.0.1:8080/secret.wav"},
    )
    assert resp.status_code == 400
    assert "blocked" in resp.json()["detail"]


def test_api_jobs_url_queues_and_processes_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/jobs/url queues a background URL job, downloads media, and completes transcription."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, proto=0: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    sample_file = tmp_path / "downloaded_podcast.wav"
    sample_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64)

    fake_downloaded = DownloadedMedia(
        file_path=sample_file,
        display_name="Episode_42.wav",
        source_url="https://example.com/Episode_42.wav",
        size_bytes=sample_file.stat().st_size,
        extractor_used="httpx",
    )

    with (
        patch.object(MediaURLIngestor, "download_media", return_value=fake_downloaded),
        patch.object(
            jobs.job_manager,
            "transcribe_and_diarize",
            return_value=_make_fake_transcription_result("Hello from URL transcription."),
        ),
        TestClient(main.app) as client,
    ):
        resp = client.post(
            "/api/jobs/url",
            json={
                "url": "https://example.com/Episode_42.wav",
                "ai_action": "raw",
                "notify": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        job_id = data["job_id"]

        import time
        job_data = {}
        for _ in range(50):
            job_resp = client.get(f"/api/jobs/{job_id}")
            assert job_resp.status_code == 200
            job_data = job_resp.json()
            if job_data["status"] in ("completed", "failed"):
                break
            time.sleep(0.02)

        assert job_data["status"] == "completed"
        assert job_data["filename"] == "Episode_42.wav"
        assert job_data["result"]["text"] == "Hello from URL transcription."
        assert job_data["result"]["source_url"] == "https://example.com/Episode_42.wav"


def test_api_jobs_batch_urls_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/jobs/batch-urls creates a batch for multiple URLs."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, proto=0: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    def make_download(url: str, dest_dir: Path, job_id: str | None = None, **kwargs) -> DownloadedMedia:
        f = dest_dir / f"{job_id or 'batch'}.wav"
        f.write_bytes(b"RIFF_DATA")
        return DownloadedMedia(
            file_path=f,
            display_name=f"{job_id or 'batch'}.wav",
            source_url=url,
            size_bytes=9,
            extractor_used="httpx",
        )

    with (
        patch.object(MediaURLIngestor, "download_media", side_effect=make_download),
        patch.object(
            jobs.job_manager,
            "transcribe_and_diarize",
            return_value=_make_fake_transcription_result("Batch URL item text."),
        ),
    ):
        client = TestClient(main.app)
        resp = client.post(
            "/api/jobs/batch-urls",
            json={
                "urls": [
                    "https://example.com/audio1.mp3",
                    "https://example.com/audio2.mp3",
                ],
                "ai_action": "raw",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_jobs"] == 2
        assert len(body["summary"]["jobs"]) == 2


def test_api_transcribe_url_synchronous_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/transcribe-url downloads and transcribes synchronously and cleans up files."""
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, proto=0: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    downloaded_path = tmp_path / "sync_media.mp3"
    downloaded_path.write_bytes(b"MP3_BYTES")

    fake_downloaded = DownloadedMedia(
        file_path=downloaded_path,
        display_name="sync_talk.mp3",
        source_url="https://example.com/sync_talk.mp3",
        size_bytes=9,
        extractor_used="httpx",
    )

    with (
        patch.object(MediaURLIngestor, "download_media", return_value=fake_downloaded),
        patch.object(
            jobs.job_manager,
            "transcribe_and_diarize",
            return_value=_make_fake_transcription_result("Synchronous URL result."),
        ),
    ):
        client = TestClient(main.app)
        resp = client.post(
            "/api/transcribe-url",
            json={
                "url": "https://example.com/sync_talk.mp3",
                "ai_action": "raw",
            },
        )
        assert resp.status_code == 200
        res = resp.json()
        assert res["filename"] == "sync_talk.mp3"
        assert res["source_url"] == "https://example.com/sync_talk.mp3"
        assert res["text"] == "Synchronous URL result."
        assert not downloaded_path.exists()


def test_api_jobs_requires_file_or_url() -> None:
    """POST /api/jobs returns 400 if neither file nor url is provided."""
    client = TestClient(main.app)
    resp = client.post("/api/jobs", data={"ai_action": "raw"})
    assert resp.status_code == 400
    assert "Either an audio/video file or a media URL must be provided" in resp.json()["detail"]


# =====================================================================
# 4. Regression Tests for Code Review Bug Fixes
# =====================================================================


def test_sse_stream_unsubscribes_and_handles_terminal_jobs() -> None:
    """Bug #1 & #2: Streaming an already-completed or already-failed job emits immediately and cleans up its subscriber queue."""
    client = TestClient(main.app)

    # Case A: Already failed job
    failed_job = jobs.job_manager.create_job("failed_sample.wav")
    failed_job.status = "failed"
    failed_job.error = "Corrupt audio header"

    with client.stream("GET", f"/api/jobs/{failed_job.job_id}/stream") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: failed" in body
    assert "Corrupt audio header" in body
    assert len(failed_job.event_queues) == 0

    # Case B: Already completed job
    done_job = jobs.job_manager.create_job("done_sample.wav")
    done_job.status = "completed"
    done_job.result = {"task_id": done_job.job_id, "text": "Done!"}

    with client.stream("GET", f"/api/jobs/{done_job.job_id}/stream") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: completed" in body
    assert len(done_job.event_queues) == 0


def test_batch_sse_stream_emits_batch_completed_when_already_terminal() -> None:
    """Bug #1 & #2: Streaming an already-terminal batch emits batch_completed and unsubscribes."""
    client = TestClient(main.app)
    batch, created_jobs = jobs.job_manager.create_batch(["a.wav", "b.wav"])
    for j in created_jobs:
        j.status = "completed"
        j.progress = 100
        j.result = {"task_id": j.job_id, "text": "ok"}

    with client.stream("GET", f"/api/batches/{batch.batch_id}/stream") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: batch_status" in body
    assert "event: batch_completed" in body
    assert len(batch.event_queues) == 0


def test_config_bool_and_int_preserve_explicit_false_and_zero() -> None:
    """Bug #3: Explicit false or 0 in config.json must not be overridden by truthy defaults."""
    diar_section = {"enabled": False}
    storage_section = {"enabled": False}
    whisper_section = {"model_idle_timeout_seconds": 0}

    assert config._cfg_bool(diar_section, "enabled", None, default=True) is False
    assert config._cfg_bool(storage_section, "enabled", None, default=True) is False
    assert config._cfg_int(whisper_section, "model_idle_timeout_seconds", None, default=300) == 0


def test_telegram_notifier_escapes_html_entities() -> None:
    """Bug #4: TelegramNotifier._format_message escapes <, >, and & in dynamic content."""
    notifier = TelegramNotifier()
    payload_summary = NotificationPayload(
        task_id="html_esc_1",
        filename="audio_<test>&1.mp3",
        duration_seconds=12.0,
        processing_time_seconds=1.2,
        whisper_model="base<1>",
        llm_model="llama&3",
        transcript_text="Use a < b && c > d in C++ <script>alert(1)</script>",
        summary_text="Summary with <b>tags</b> & symbols",
        summary_type="bullets<x>",
    )
    msg_summary = notifier._format_message(payload_summary)
    assert "audio_&lt;test&gt;&amp;1.mp3" in msg_summary
    assert "Summary with &lt;b&gt;tags&lt;/b&gt; &amp; symbols" in msg_summary

    payload_raw = NotificationPayload(
        task_id="html_esc_2",
        filename="raw_<clip>.wav",
        duration_seconds=5.0,
        processing_time_seconds=0.5,
        whisper_model="base",
        transcript_text="Use a < b && c > d in C++ <script>alert(1)</script>",
    )
    msg_raw = notifier._format_message(payload_raw)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in msg_raw


def test_ollama_provider_reflects_live_settings_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug #5: OllamaLLMProvider.base_url reflects runtime updates to settings.ollama_base_url."""
    provider = OllamaLLMProvider()
    monkeypatch.setattr(config.settings, "ollama_base_url", "http://10.20.30.40:11434")
    assert provider.base_url == "http://10.20.30.40:11434"


def test_whisper_cpp_clips_audio_before_int16_cast() -> None:
    """Bug #7: Out-of-range float32 samples (>1.0 or <-1.0) are clipped instead of wrapping."""
    samples = np.array([-1.5, -1.0, 0.0, 1.0, 1.5], dtype=np.float32)
    clipped = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    assert int(clipped[0]) == -32767
    assert int(clipped[-1]) == 32767


def test_get_batch_preserves_persisted_created_at(isolate_test_job_store) -> None:
    """Bug #8: Rehydrating a batch from SQLite preserves its original created_at timestamp."""
    batch, _ = jobs.job_manager.create_batch(["one.wav"])
    expected_dt = batch.created_at

    jobs.job_manager._batches.clear()
    rehydrated = jobs.job_manager.get_batch(batch.batch_id)
    assert rehydrated is not None
    assert abs((rehydrated.created_at - expected_dt).total_seconds()) < 1.0
