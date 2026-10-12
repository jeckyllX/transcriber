"""Tests for Background Job Manager and streaming event queues."""

import pytest
from jobs import JobManager, Job


def test_job_lifecycle():
    manager = JobManager()
    job = manager.create_job("test_audio.mp3")

    assert job.job_id is not None
    assert job.filename == "test_audio.mp3"
    assert job.status == "queued"
    assert manager.get_job(job.job_id) is job

    # Status update
    manager.update_status(job.job_id, "converting", 25, "Decoding audio...")
    assert job.status == "converting"
    assert job.progress == 25
    assert job.message == "Decoding audio..."


@pytest.mark.asyncio
async def test_job_event_subscription():
    manager = JobManager()
    job = manager.create_job("test_stream.wav")

    queue = manager.subscribe(job.job_id)
    assert len(job.event_queues) == 1

    manager.emit(job.job_id, "segment", {"text": "Hello world", "start": 0.0, "end": 2.0})
    event = await queue.get()

    assert event["event"] == "segment"
    assert event["data"]["text"] == "Hello world"

    manager.unsubscribe(job.job_id, queue)
    assert len(job.event_queues) == 0


@pytest.mark.asyncio
async def test_batch_job_lifecycle():
    manager = JobManager()
    batch, jobs = manager.create_batch(["first.wav", "second.wav"])

    assert batch.batch_id is not None
    assert len(jobs) == 2
    assert jobs[0].batch_id == batch.batch_id
    assert jobs[1].batch_id == batch.batch_id
    assert manager.get_batch(batch.batch_id) is batch

    summary = manager.get_batch_summary(batch.batch_id)
    assert summary is not None
    assert summary["total_jobs"] == 2
    assert summary["queued_jobs"] == 2
    assert summary["completed_jobs"] == 0
    assert summary["overall_progress"] == 0
    assert summary["status"] == "queued"

    # Update first job to processing
    manager.update_status(jobs[0].job_id, "transcribing", 50, "Transcribing segment 1...")
    summary = manager.get_batch_summary(batch.batch_id)
    assert summary["status"] == "processing"
    assert summary["overall_progress"] == 25  # (50 + 0) // 2

    # Complete both jobs
    manager.update_status(jobs[0].job_id, "completed", 100, "Done")
    manager.update_status(jobs[1].job_id, "completed", 100, "Done")
    summary = manager.get_batch_summary(batch.batch_id)
    assert summary["status"] == "completed"
    assert summary["completed_jobs"] == 2
    assert summary["overall_progress"] == 100

    # Test event subscription for batch
    queue = manager.subscribe_batch(batch.batch_id)
    assert len(batch.event_queues) == 1

    manager.emit_batch(batch.batch_id, "batch_completed", {"batch_id": batch.batch_id})
    event = await queue.get()
    assert event["event"] == "batch_completed"
    assert event["data"]["batch_id"] == batch.batch_id

    manager.unsubscribe_batch(batch.batch_id, queue)
    assert len(batch.event_queues) == 0


def test_batch_zip_generation():
    import io
    import json
    import zipfile

    manager = JobManager()
    batch, jobs = manager.create_batch(["interview.mp3", "notes.wav"])
    job1, job2 = jobs[0], jobs[1]

    job1.status = "completed"
    job1.duration = 45.2
    job1.processing_time = 3.5
    job1.result = {
        "text": "Hello, welcome to the interview.",
        "srt": "1\n00:00:00,000 --> 00:00:02,000\nHello, welcome to the interview.",
        "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello, welcome to the interview.",
        "ass": "[Script Info]\nTitle: Subtitle",
        "polished": "Hello, welcome to the interview!",
        "summary": "Introduction to the interview.",
        "engine_used": "faster-whisper",
        "model_used": "tiny",
    }

    # Job2 remains without result
    zip_bytes = manager.generate_batch_zip(batch.batch_id)
    assert zip_bytes is not None

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "batch_manifest.json" in namelist

        manifest_data = json.loads(zf.read("batch_manifest.json").decode("utf-8"))
        assert manifest_data["batch_id"] == batch.batch_id
        assert manifest_data["total_files"] == 2
        assert manifest_data["exported_files"] == 1
        assert manifest_data["files"][0]["job_id"] == job1.job_id

        # Verify export files exist for job1
        expected_folder = f"interview_{job1.job_id}"
        assert f"{expected_folder}/interview.txt" in namelist
        assert f"{expected_folder}/interview.srt" in namelist
        assert f"{expected_folder}/interview.vtt" in namelist
        assert f"{expected_folder}/interview.ass" in namelist
        assert f"{expected_folder}/interview.json" in namelist
        assert f"{expected_folder}/interview_polished.txt" in namelist
        assert f"{expected_folder}/interview_summary.txt" in namelist

        assert zf.read(f"{expected_folder}/interview.txt").decode("utf-8") == "Hello, welcome to the interview."

    # Non-existent batch returns None
    assert manager.generate_batch_zip("non_existent_id") is None

