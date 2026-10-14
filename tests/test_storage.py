"""Unit tests for SQLite persistent JobStore with Full-Text Search (FTS5)."""

import pytest
from datetime import datetime, timezone
from storage import JobStore
from jobs import Job


@pytest.fixture
def temp_store(tmp_path):
    db_file = tmp_path / "test_transcriptions.db"
    store = JobStore(db_path=db_file, enabled=True)
    return store


def test_job_store_initialization(temp_store):
    assert temp_store.db_path.exists()
    stats = temp_store.get_stats()
    assert stats["enabled"] is True
    assert stats["total_jobs"] == 0


def test_job_store_save_and_get(temp_store):
    job = Job(
        job_id="test001",
        filename="meeting_audio.mp3",
        status="completed",
        progress=100,
        message="Done",
        duration=120.5,
        processing_time=12.3,
        result={
            "text": "Welcome to our product roadmap review.",
            "language": "en",
            "engine_used": "faster-whisper",
            "model_used": "base",
            "num_speakers": 2,
            "audio_enhanced": True,
            "summary": "Roadmap discussion and milestone planning.",
            "summaries": {
                "bullets": "Roadmap discussion and milestone planning.",
                "tldr": "Product roadmap review.",
            },
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Welcome to our product roadmap review.",
                    "speaker": "Speaker 0",
                    "words": [],
                }
            ],
            "speaker_turns": [],
        },
    )

    temp_store.save_job(job)

    loaded = temp_store.get_job("test001")
    assert loaded is not None
    assert loaded.job_id == "test001"
    assert loaded.filename == "meeting_audio.mp3"
    assert loaded.status == "completed"
    assert loaded.duration == 120.5
    assert loaded.result is not None
    assert loaded.result["engine_used"] == "faster-whisper"
    assert loaded.result["summaries"]["tldr"] == "Product roadmap review."
    assert loaded.result["segments"][0]["speaker"] == "Speaker 0"


def test_job_store_pagination_and_filtering(temp_store):
    for i in range(5):
        status = "completed" if i % 2 == 0 else "failed"
        job = Job(
            job_id=f"job_{i}",
            filename=f"audio_{i}.wav",
            status=status,
            progress=100 if status == "completed" else 0,
            message="Status message",
            duration=10.0 * (i + 1),
            result={"text": f"Transcription number {i}"} if status == "completed" else None,
            created_at=datetime.now(timezone.utc),
        )
        temp_store.save_job(job)

    # Test pagination
    items, total = temp_store.list_jobs(page=1, page_size=2)
    assert total == 5
    assert len(items) == 2

    # Test second page
    page2_items, _ = temp_store.list_jobs(page=2, page_size=2)
    assert len(page2_items) == 2
    assert page2_items[0]["job_id"] != items[0]["job_id"]

    # Test filtering by status
    completed_items, comp_total = temp_store.list_jobs(status="completed")
    assert comp_total == 3
    assert all(it["status"] == "completed" for it in completed_items)

    failed_items, fail_total = temp_store.list_jobs(status="failed")
    assert fail_total == 2
    assert all(it["status"] == "failed" for it in failed_items)


def test_job_store_fts_search(temp_store):
    job1 = Job(
        job_id="job_fin",
        filename="financial_q3.wav",
        status="completed",
        progress=100,
        result={
            "text": "The quarterly financial revenue exceeded all initial forecasts by twenty percent.",
            "summary": "Executive briefing on record revenue and profit margins.",
        },
    )

    job2 = Job(
        job_id="job_ux",
        filename="design_sprint.m4a",
        status="completed",
        progress=100,
        result={
            "text": "We finalized the onboarding wireframes and mobile navigation components.",
            "summary": "UX sprint summary focusing on user retention and mobile layouts.",
        },
    )

    temp_store.save_job(job1)
    temp_store.save_job(job2)

    # Search in text
    res_fin = temp_store.search_transcripts("revenue")
    assert len(res_fin) == 1
    assert res_fin[0]["job_id"] == "job_fin"
    assert "<mark>" in res_fin[0]["text_snippet"] or "revenue" in res_fin[0]["text_snippet"]

    # Search in summary
    res_ux = temp_store.search_transcripts("retention")
    assert len(res_ux) == 1
    assert res_ux[0]["job_id"] == "job_ux"

    # Search non-existent
    res_none = temp_store.search_transcripts("quantum physics")
    assert len(res_none) == 0


def test_job_store_delete(temp_store):
    job = Job(
        job_id="to_delete",
        filename="temp.wav",
        status="completed",
        result={"text": "Temporary secret audio that will be erased."},
    )
    temp_store.save_job(job)

    # Verify present in search
    assert len(temp_store.search_transcripts("secret")) == 1

    # Delete
    deleted = temp_store.delete_job("to_delete")
    assert deleted is True

    # Verify gone from retrieval and search
    assert temp_store.get_job("to_delete") is None
    assert len(temp_store.search_transcripts("secret")) == 0

    # Delete nonexistent returns False
    assert temp_store.delete_job("nonexistent") is False


def test_job_store_batches(temp_store):
    temp_store.save_batch("batch_alpha", total_files=3, status="processing", progress=50)

    # Add jobs linked to batch
    for i in range(3):
        job = Job(
            job_id=f"bjob_{i}",
            batch_id="batch_alpha",
            filename=f"file_{i}.wav",
            status="completed" if i < 2 else "processing",
        )
        temp_store.save_job(job)

    batch_data = temp_store.get_batch("batch_alpha")
    assert batch_data is not None
    assert batch_data["batch_id"] == "batch_alpha"
    assert batch_data["total_files"] == 3
    assert len(batch_data["jobs"]) == 3

    # Delete batch with jobs
    deleted = temp_store.delete_batch("batch_alpha", delete_jobs=True)
    assert deleted is True
    assert temp_store.get_batch("batch_alpha") is None
    assert temp_store.get_job("bjob_0") is None
