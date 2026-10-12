"""Asynchronous Background Job Queue & Event Streaming.

Allows long audio files to be processed without blocking HTTP connections,
streaming real-time progress and transcribed segments via Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from audio_processor import AudioProcessor
from notifications import dispatcher, NotificationPayload
from transcribers import transcriber_factory
from transcribers.base import Segment
from llm import llm_registry, get_polish_prompt, ChunkedSummarizer

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents a background audio processing task."""
    job_id: str
    filename: str
    status: str = "queued"  # queued, converting, transcribing, processing_ai, completed, failed
    progress: int = 0
    message: str = "Job queued"
    duration: float = 0.0
    processing_time: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_queues: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    batch_id: str | None = None


@dataclass
class BatchJob:
    """Represents a collection of jobs submitted together for sequential or concurrent processing."""
    batch_id: str
    job_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_queues: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)


class JobManager:
    """In-memory job coordinator with live event fan-out."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._batches: dict[str, BatchJob] = {}
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            from config import settings
            self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        return self._semaphore

    def create_job(self, filename: str, batch_id: str | None = None) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, filename=filename, batch_id=batch_id)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def create_batch(self, filenames: list[str]) -> tuple[BatchJob, list[Job]]:
        batch_id = str(uuid.uuid4())[:8]
        batch = BatchJob(batch_id=batch_id)
        created_jobs = []
        for filename in filenames:
            job = self.create_job(filename, batch_id=batch_id)
            batch.job_ids.append(job.job_id)
            created_jobs.append(job)
        self._batches[batch_id] = batch
        return batch, created_jobs

    def get_batch(self, batch_id: str) -> BatchJob | None:
        return self._batches.get(batch_id)

    def list_batches(self) -> list[dict[str, Any]]:
        return [
            summary for b_id in self._batches
            if (summary := self.get_batch_summary(b_id)) is not None
        ]

    def get_batch_summary(self, batch_id: str) -> dict[str, Any] | None:
        batch = self.get_batch(batch_id)
        if not batch:
            return None

        jobs = [self.get_job(jid) for jid in batch.job_ids]
        jobs = [j for j in jobs if j is not None]

        total = len(jobs)
        completed = sum(1 for j in jobs if j.status == "completed")
        failed = sum(1 for j in jobs if j.status == "failed")
        processing = sum(1 for j in jobs if j.status not in ("queued", "completed", "failed"))
        queued = sum(1 for j in jobs if j.status == "queued")

        if total == 0:
            overall_status = "empty"
            overall_progress = 100
        elif completed == total:
            overall_status = "completed"
            overall_progress = 100
        elif completed + failed == total:
            overall_status = "partial_failure" if (failed > 0 and completed > 0) else "failed"
            overall_progress = 100
        elif processing > 0 or completed > 0:
            overall_status = "processing"
            overall_progress = sum(j.progress for j in jobs) // total if total else 0
        else:
            overall_status = "queued"
            overall_progress = 0

        return {
            "batch_id": batch_id,
            "created_at": batch.created_at.isoformat(),
            "status": overall_status,
            "total_jobs": total,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "processing_jobs": processing,
            "queued_jobs": queued,
            "overall_progress": overall_progress,
            "jobs": [
                {
                    "job_id": j.job_id,
                    "filename": j.filename,
                    "status": j.status,
                    "progress": j.progress,
                    "message": j.message,
                    "duration": j.duration,
                    "processing_time": j.processing_time,
                    "error": j.error,
                    "has_result": j.result is not None,
                }
                for j in jobs
            ],
        }

    def subscribe_batch(self, batch_id: str) -> asyncio.Queue[dict[str, Any]]:
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found")
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        batch.event_queues.append(q)
        return q

    def unsubscribe_batch(self, batch_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        batch = self.get_batch(batch_id)
        if batch and queue in batch.event_queues:
            batch.event_queues.remove(queue)

    def emit_batch(self, batch_id: str, event: str, data: dict[str, Any]) -> None:
        batch = self.get_batch(batch_id)
        if not batch:
            return
        payload = {"event": event, "data": data, "timestamp": time.time()}
        for q in list(batch.event_queues):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    def generate_batch_zip(self, batch_id: str) -> bytes | None:
        batch = self.get_batch(batch_id)
        if not batch:
            return None

        import io
        import json
        import zipfile

        buf = io.BytesIO()
        manifest_entries = []

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for jid in batch.job_ids:
                job = self.get_job(jid)
                if not job or not job.result:
                    continue

                base_name = Path(job.filename).stem or f"job_{job.job_id}"
                safe_name = "".join(c for c in base_name if c.isalnum() or c in ("-", "_", " ")).strip() or f"job_{job.job_id}"
                folder = f"{safe_name}_{job.job_id}"

                if job.result.get("text"):
                    zf.writestr(f"{folder}/{safe_name}.txt", job.result["text"])
                if job.result.get("srt"):
                    zf.writestr(f"{folder}/{safe_name}.srt", job.result["srt"])
                if job.result.get("vtt"):
                    zf.writestr(f"{folder}/{safe_name}.vtt", job.result["vtt"])
                if job.result.get("ass"):
                    zf.writestr(f"{folder}/{safe_name}.ass", job.result["ass"])
                zf.writestr(f"{folder}/{safe_name}.json", json.dumps(job.result, indent=2))
                if job.result.get("polished"):
                    zf.writestr(f"{folder}/{safe_name}_polished.txt", job.result["polished"])
                if job.result.get("summary"):
                    zf.writestr(f"{folder}/{safe_name}_summary.txt", job.result["summary"])

                manifest_entries.append({
                    "job_id": job.job_id,
                    "filename": job.filename,
                    "duration": job.duration,
                    "processing_time": job.processing_time,
                    "engine_used": job.result.get("engine_used"),
                    "model_used": job.result.get("model_used"),
                    "status": job.status,
                })

            zf.writestr(
                "batch_manifest.json",
                json.dumps({
                    "batch_id": batch_id,
                    "created_at": batch.created_at.isoformat(),
                    "total_files": len(batch.job_ids),
                    "exported_files": len(manifest_entries),
                    "files": manifest_entries,
                }, indent=2)
            )

        buf.seek(0)
        return buf.getvalue()


    def rename_speaker(self, job_id: str, old_name: str, new_name: str) -> dict[str, Any] | None:
        """Rename a detected speaker across all segments, turns, and subtitle exports."""
        job = self.get_job(job_id)
        if not job or not job.result:
            return None

        segments_data = job.result.get("segments", [])
        for s in segments_data:
            if s.get("speaker") == old_name:
                s["speaker"] = new_name
            for w in s.get("words", []):
                if w.get("speaker") == old_name:
                    w["speaker"] = new_name

        from transcribers.base import Segment, TranscriptionResult
        segments = [Segment(**s) for s in segments_data]
        res = TranscriptionResult(
            text=job.result.get("text", ""),
            segments=segments,
            language=job.result.get("language", "auto"),
            duration=job.result.get("duration", 0.0),
        )
        job.result["text"] = res.to_txt()
        job.result["srt"] = res.to_srt()
        job.result["vtt"] = res.to_vtt()
        job.result["ass"] = res.to_ass()
        job.result["word_vtt"] = res.to_word_vtt()
        job.result["segments"] = [s.model_dump() for s in segments]
        job.result["speaker_turns"] = res.get_speaker_turns()

        self.emit(job_id, "speaker_renamed", {
            "old_name": old_name,
            "new_name": new_name,
            "speaker_turns": job.result["speaker_turns"],
        })
        return job.result

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.event_queues.append(q)
        return q

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        job = self.get_job(job_id)
        if job and queue in job.event_queues:
            job.event_queues.remove(queue)

    def emit(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        payload = {"event": event, "data": data, "timestamp": time.time()}
        for q in list(job.event_queues):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

        if job.batch_id:
            summary = self.get_batch_summary(job.batch_id)
            if summary:
                self.emit_batch(
                    job.batch_id,
                    "batch_progress",
                    {"job_id": job_id, "event": event, "data": data, "summary": summary},
                )


    def update_status(self, job_id: str, status: str, progress: int, message: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = status
        job.progress = progress
        job.message = message
        self.emit(job_id, "status", {"status": status, "progress": progress, "message": message})

    async def run_pipeline(
        self,
        job_id: str,
        media_path: Path,
        whisper_engine: str = "faster-whisper",
        whisper_model: str = "base",
        language: str | None = None,
        vad_filter: bool = True,
        enable_diarization: bool = False,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        ai_action: str = "summary",  # raw, polish, summary
        summary_level: str = "bullets",
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        notify: bool = False,
        clean_audio: bool = False,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        start_time = time.time()
        audio_processor = AudioProcessor()
        converted_wav: Path | None = None

        try:
            async with self.semaphore:
                # 1. Convert audio
                if clean_audio:
                    self.update_status(
                        job_id,
                        "converting",
                        15,
                        "Converting and enhancing audio (noise reduction & leveling)...",
                    )
                else:
                    self.update_status(job_id, "converting", 15, "Decoding audio with FFmpeg...")
                transcriber = transcriber_factory.get_transcriber(whisper_engine)

                from config import settings
                from audio_processor import HAS_NUMPY

                use_memory = settings.use_in_memory_pcm and HAS_NUMPY and whisper_engine == "faster-whisper"

                if use_memory:
                    audio_input, metadata = audio_processor.convert_to_pcm_array(
                        media_path, enhance_audio=clean_audio
                    )
                else:
                    converted_wav = media_path.parent / f"{job_id}_16k.wav"
                    audio_input = audio_processor.convert_to_whisper_wav(
                        media_path, converted_wav, enhance_audio=clean_audio
                    )
                    metadata = audio_processor.probe_media(media_path)

                job.duration = metadata.duration

                # 2. Transcribe with live segment streaming
                self.update_status(job_id, "transcribing", 35, "Transcribing with Whisper...")

                def on_segment_callback(seg: Segment):
                    self.emit(job_id, "segment", seg.model_dump())

                # Run CPU-bound transcription in threadpool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                transcription_result = await loop.run_in_executor(
                    None,
                    lambda: transcriber.transcribe(
                        audio_input,
                        model_name=whisper_model,
                        language=language if language != "auto" else None,
                        vad_filter=vad_filter,
                        on_segment=on_segment_callback,
                    ),
                )

                # 2.5 Speaker Diarization
                num_speakers_detected = 0
                if enable_diarization:
                    try:
                        from diarization import diarizer_factory, align_speakers_to_segments
                        diarizer = diarizer_factory.get_diarizer("sherpa-onnx")
                        if diarizer.is_available():
                            if hasattr(diarizer, "models_ready") and not diarizer.models_ready():
                                self.update_status(job_id, "diarizing", 52, "Downloading speaker diarization models on demand (first run only)...")
                            else:
                                self.update_status(job_id, "diarizing", 60, "Identifying speakers (diarization)...")
                            diar_result = await loop.run_in_executor(
                                None,
                                lambda: diarizer.diarize(
                                    audio_input,
                                    num_speakers=num_speakers,
                                    cluster_threshold=cluster_threshold,
                                ),
                            )
                            align_speakers_to_segments(transcription_result.segments, diar_result)
                            num_speakers_detected = diar_result.num_speakers
                            self.emit(job_id, "diarization", {
                                "num_speakers": diar_result.num_speakers,
                                "speakers": diar_result.speaker_names,
                                "intervals": [i.model_dump() for i in diar_result.intervals],
                            })
                        else:
                            logger.warning("Speaker diarization requested but engine or models are not available; skipping.")
                    except Exception as diar_err:
                        logger.warning("Diarization failed for job %s: %s", job_id, diar_err)

            job_result: dict[str, Any] = {
                "task_id": job_id,
                "filename": job.filename,
                "duration": job.duration or transcription_result.duration,
                "language": transcription_result.language,
                "engine_used": transcriber.name,
                "model_used": whisper_model,
                "num_speakers": num_speakers_detected,
                "speaker_turns": transcription_result.get_speaker_turns(),
                "text": transcription_result.to_txt(),
                "srt": transcription_result.to_srt(),
                "vtt": transcription_result.to_vtt(),
                "ass": transcription_result.to_ass(),
                "word_vtt": transcription_result.to_word_vtt(),
                "segments": [s.model_dump() for s in transcription_result.segments],
                "polished": None,
                "summary": None,
                "audio_enhanced": clean_audio,
            }

            # 3. AI Post-Processing (gracefully degraded if LLM provider is offline)
            if ai_action in ("polish", "summary") and transcription_result.text.strip():
                self.update_status(job_id, "processing_ai", 75, f"Running AI {ai_action}...")
                try:
                    provider = llm_registry.get_provider(llm_provider)

                    if ai_action == "polish":
                        sys_prompt, user_prompt = get_polish_prompt(transcription_result.text)
                        accumulated = []
                        async for token in provider.generate_stream(
                            prompt=user_prompt, system_prompt=sys_prompt, model=llm_model
                        ):
                            accumulated.append(token)
                            self.emit(job_id, "ai_token", {"token": token, "action": "polish"})
                        job_result["polished"] = "".join(accumulated)

                    elif ai_action == "summary":
                        summarizer = ChunkedSummarizer(provider)
                        accumulated = []
                        async for token in summarizer.summarize_stream(
                            transcript=transcription_result.text,
                            level=summary_level,
                            model=llm_model,
                        ):
                            accumulated.append(token)
                            self.emit(job_id, "ai_token", {"token": token, "action": "summary"})
                        job_result["summary"] = "".join(accumulated)
                except Exception as ai_err:
                    logger.warning("AI post-processing (%s) failed for job %s: %s", ai_action, job_id, ai_err)
                    warn_text = f"AI {ai_action} skipped: {ai_err}. Check that Ollama is running ('ollama serve') or your API key is valid."
                    job_result["ai_warning"] = warn_text
                    if ai_action == "summary":
                        job_result["summary"] = f"[{warn_text}]"
                    elif ai_action == "polish":
                        job_result["polished"] = f"[{warn_text}]"
                    self.emit(job_id, "status", {"state": "warning", "progress": 85, "message": warn_text})

            # 4. Optional Notification Dispatch
            if notify:
                self.update_status(job_id, "dispatching_notification", 95, "Dispatching notifications...")
                payload = NotificationPayload(
                    task_id=job_id,
                    filename=job.filename,
                    duration_seconds=job_result["duration"],
                    processing_time_seconds=round(time.time() - start_time, 2),
                    whisper_model=whisper_model,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    summary_type=ai_action,
                    transcript_text=job_result["text"],
                    summary_text=job_result["summary"],
                    polished_text=job_result["polished"],
                )
                await dispatcher.dispatch_all(payload)

            elapsed = round(time.time() - start_time, 2)
            job.processing_time = elapsed
            job_result["processing_time"] = elapsed
            job.result = job_result

            self.update_status(job_id, "completed", 100, f"Completed in {elapsed}s")
            self.emit(job_id, "completed", job_result)

        except Exception as e:
            logger.exception("Pipeline job %s failed", job_id)
            job.status = "failed"
            job.error = str(e)
            self.update_status(job_id, "failed", 0, str(e))
            self.emit(job_id, "failed", {"error": str(e)})

        finally:
            if converted_wav and converted_wav.exists():
                try:
                    converted_wav.unlink()
                except Exception:
                    pass
            # Clean up source file
            if media_path.exists():
                try:
                    media_path.unlink()
                except Exception:
                    pass


# Global job manager instance
job_manager = JobManager()
