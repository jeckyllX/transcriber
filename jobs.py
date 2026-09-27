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


class JobManager:
    """In-memory job coordinator with live event fan-out."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self, filename: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, filename=filename)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

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
        ai_action: str = "summary",  # raw, polish, summary
        summary_level: str = "bullets",
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        notify: bool = False,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        start_time = time.time()
        audio_processor = AudioProcessor()

        try:
            # 1. Convert audio
            self.update_status(job_id, "converting", 15, "Decoding audio with FFmpeg...")
            transcriber = transcriber_factory.get_transcriber(whisper_engine)

            from config import settings
            from audio_processor import HAS_NUMPY

            use_memory = settings.use_in_memory_pcm and HAS_NUMPY and whisper_engine == "faster-whisper"

            if use_memory:
                audio_input, metadata = audio_processor.convert_to_pcm_array(media_path)
            else:
                converted_wav = media_path.parent / f"{job_id}_16k.wav"
                audio_input = audio_processor.convert_to_whisper_wav(media_path, converted_wav)
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

            job_result: dict[str, Any] = {
                "task_id": job_id,
                "filename": job.filename,
                "duration": job.duration or transcription_result.duration,
                "language": transcription_result.language,
                "engine_used": transcriber.name,
                "model_used": whisper_model,
                "text": transcription_result.to_txt(),
                "srt": transcription_result.to_srt(),
                "vtt": transcription_result.to_vtt(),
                "ass": transcription_result.to_ass(),
                "word_vtt": transcription_result.to_word_vtt(),
                "segments": [s.model_dump() for s in transcription_result.segments],
                "polished": None,
                "summary": None,
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
            # Clean up source file
            if media_path.exists():
                try:
                    media_path.unlink()
                except Exception:
                    pass


# Global job manager instance
job_manager = JobManager()
