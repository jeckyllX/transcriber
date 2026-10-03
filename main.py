"""Main FastAPI application module.

Provides REST and Server-Sent Events (SSE) streaming endpoints for audio transcription,
background job management, LLM polishing/summarization, model management, and notification dispatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx

from config import BASE_DIR, STATIC_DIR, TEMPLATES_DIR, UPLOADS_DIR, settings, save_config, get_masked_settings
from audio_processor import AudioProcessor, AudioProcessorError
from transcribers import transcriber_factory
from transcribers.streaming import LiveTranscriptionSession
from llm import llm_registry, get_polish_prompt, get_summary_prompt, ChunkedSummarizer
from notifications import dispatcher, NotificationPayload, TelegramNotifier, WebhookNotifier
from jobs import job_manager
from diarization import diarizer_factory, align_speakers_to_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("transcriber")

app = FastAPI(
    title="Audio Transcriber & AI Summarizer",
    description="Audio transcription with Whisper, Ollama polishing/summarization, and notifications.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
audio_processor = AudioProcessor()


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class LLMProcessRequest(BaseModel):
    text: str
    action: str = "summary"  # 'polish' or 'summary'
    detail_level: str = "bullets"  # 'tldr', 'bullets', 'detailed', 'action_items', 'custom'
    custom_instruction: str | None = None
    provider: str = "ollama"
    model: str | None = None


class SettingsTestRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None


class NotificationTestRequest(BaseModel):
    provider: str
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None


class DispatchNotificationRequest(BaseModel):
    task_id: str
    filename: str
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    whisper_model: str = "base"
    llm_provider: str | None = None
    llm_model: str | None = None
    summary_type: str | None = None
    transcript_text: str = ""
    summary_text: str | None = None
    polished_text: str | None = None


class RenameSpeakerRequest(BaseModel):
    old_name: str
    new_name: str


# ------------------------------------------------------------------------------
# UI Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_whisper_model": settings.default_whisper_model,
            "default_ollama_model": settings.default_ollama_model,
        }
    )


# ------------------------------------------------------------------------------
# System Diagnostics & Model Management
# ------------------------------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    ollama_prov = llm_registry.get_provider("ollama")
    ollama_status = await ollama_prov.test_connection()

    return {
        "status": "healthy",
        "ffmpeg": {
            "available": bool(settings.ffmpeg_bin),
            "binary_path": settings.ffmpeg_bin,
        },
        "whisper_engines": transcriber_factory.list_engines(),
        "diarization_engines": diarizer_factory.list_engines(),
        "llm_providers": llm_registry.list_providers(),
        "ollama_connection": ollama_status,
        "notification_providers": dispatcher.list_providers(),
    }


@app.get("/api/llm/models")
async def get_llm_models(provider: str = "ollama"):
    try:
        prov = llm_registry.get_provider(provider)
        models = await prov.list_models()
        return {"provider": provider, "models": models}
    except Exception as e:
        logger.error("Failed to list models for provider %s: %s", provider, e)
        return {"provider": provider, "models": [], "error": str(e)}


@app.post("/api/llm/pull")
async def pull_ollama_model(model_name: str = Form(...)):
    prov = llm_registry.get_provider("ollama")
    if not hasattr(prov, "pull_model_stream"):
        raise HTTPException(status_code=400, detail="Provider does not support pulling models.")

    async def event_generator():
        async for progress in prov.pull_model_stream(model_name):
            yield f"data: {json.dumps(progress)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/whisper/models")
async def get_whisper_models(engine: str = "faster-whisper"):
    """Return supported models for the requested Whisper engine."""
    models = transcriber_factory.get_models_for_engine(engine)
    return {"engine": engine, "models": models}


@app.get("/api/settings")
async def get_settings_endpoint():
    """Retrieve runtime settings with sensitive credentials masked."""
    return get_masked_settings()


@app.post("/api/settings")
async def update_settings_endpoint(updates: dict[str, Any]):
    """Update settings and persist to config.json."""
    try:
        updated = save_config(updates)
        return {"status": "ok", "settings": updated}
    except Exception as e:
        logger.error("Failed to save settings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings/test")
async def test_provider_settings_endpoint(req: SettingsTestRequest):
    """Test connection and latency to local or cloud AI providers."""
    prov_id = req.provider

    if prov_id == "ollama":
        prov = llm_registry.get_provider("ollama")
        return await prov.test_connection()

    elif prov_id in ("groq", "openrouter", "openai", "openai_compat"):
        api_key = req.api_key
        if not api_key or "••••" in api_key:
            if prov_id == "groq":
                api_key = settings.groq_api_key
            elif prov_id == "openrouter":
                api_key = settings.openrouter_api_key
            elif prov_id == "openai":
                api_key = settings.openai_api_key
            elif prov_id == "openai_compat":
                api_key = settings.openai_api_key

        base_url = req.base_url
        if not base_url:
            if prov_id == "groq":
                base_url = settings.groq_base_url
            elif prov_id == "openrouter":
                base_url = settings.openrouter_base_url
            elif prov_id == "openai":
                base_url = settings.openai_base_url
            elif prov_id == "openai_compat":
                base_url = settings.openai_base_url

        if not api_key:
            return {"online": False, "error": f"{prov_id.capitalize()} API Key is missing or not configured."}

        extra_headers = {}
        if prov_id == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://github.com/jeckyllX/transcriber",
                "X-Title": "Transcriber",
            }

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {api_key}", **extra_headers},
                )
                latency_ms = round((time.time() - start_time) * 1000, 1)
                return {
                    "online": resp.status_code == 200,
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "base_url": base_url,
                    "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}: {resp.text[:120]}",
                }
        except Exception as err:
            return {"online": False, "base_url": base_url, "error": str(err)}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported test provider: {prov_id}")


# ------------------------------------------------------------------------------
# Asynchronous Background Job Endpoints
# ------------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_background_job(
    file: UploadFile = File(...),
    whisper_engine: str = Form(default="faster-whisper"),
    whisper_model: str = Form(default="base"),
    language: str = Form(default="auto"),
    vad_filter: bool = Form(default=True),
    enable_diarization: bool = Form(default=True),
    num_speakers: int = Form(default=-1),
    cluster_threshold: float = Form(default=0.5),
    ai_action: str = Form(default="summary"),
    summary_level: str = Form(default="bullets"),
    llm_provider: str = Form(default="ollama"),
    llm_model: str | None = Form(default=None),
    notify: bool = Form(default=False),
):
    """Submit an audio file for asynchronous processing.

    Returns immediately with job_id so clients can track real-time progress.
    """
    job = job_manager.create_job(file.filename)
    ext = Path(file.filename).suffix or ".wav"
    temp_upload = UPLOADS_DIR / f"{job.job_id}_upload{ext}"

    with open(temp_upload, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Launch background task
    asyncio.create_task(
        job_manager.run_pipeline(
            job_id=job.job_id,
            media_path=temp_upload,
            whisper_engine=whisper_engine,
            whisper_model=whisper_model,
            language=language,
            vad_filter=vad_filter,
            enable_diarization=enable_diarization,
            num_speakers=num_speakers,
            cluster_threshold=cluster_threshold,
            ai_action=ai_action,
            summary_level=summary_level,
            llm_provider=llm_provider,
            llm_model=llm_model,
            notify=notify,
        )
    )

    return {"job_id": job.job_id, "status": "queued"}


@app.post("/api/jobs/{job_id}/rename-speaker")
async def rename_job_speaker(job_id: str, req: RenameSpeakerRequest):
    """Rename a speaker in an existing job and refresh exports and dialogue turns."""
    result = job_manager.rename_speaker(job_id, req.old_name, req.new_name)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found or result not available")
    return {"status": "ok", "result": result}


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Retrieve current state and result of a job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "duration": job.duration,
        "processing_time": job.processing_time,
        "result": job.result,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(job_id: str):
    """Stream real-time job events via Server-Sent Events (SSE)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = job_manager.subscribe(job_id)

    async def sse_generator():
        # Emit initial current status
        yield f"event: status\ndata: {json.dumps({'status': job.status, 'progress': job.progress, 'message': job.message})}\n\n"

        if job.result:
            yield f"event: completed\ndata: {json.dumps(job.result)}\n\n"
            return

        try:
            while True:
                msg = await queue.get()
                event_name = msg.get("event", "message")
                data_json = json.dumps(msg.get("data", {}))
                yield f"event: {event_name}\ndata: {data_json}\n\n"

                if event_name in ("completed", "failed"):
                    break
        finally:
            job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ------------------------------------------------------------------------------
# Real-Time WebSocket Streaming Endpoint
# ------------------------------------------------------------------------------
@app.websocket("/api/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """Real-time live streaming audio transcription via WebSocket.

    Accepts binary 16kHz 16-bit mono PCM chunks and optional JSON control messages.
    Streams live interim 'partial' tokens and finalized 'segment' events.
    """
    await websocket.accept()

    session = LiveTranscriptionSession(
        whisper_engine=getattr(settings, "default_whisper_engine", "faster-whisper"),
        whisper_model=settings.default_whisper_model,
        language="auto",
    )

    is_running = True
    processing_lock = asyncio.Lock()

    async def process_loop():
        """Background loop that periodically evaluates buffer and sends recognition events."""
        while is_running:
            try:
                await asyncio.sleep(0.3)
                if not is_running:
                    break
                if session.should_process():
                    async with processing_lock:
                        events = await asyncio.to_thread(session.step)
                    for ev in events:
                        await websocket.send_json(ev)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Streaming processing loop error: %s", exc)

    proc_task = asyncio.create_task(process_loop())

    try:
        # Send ready event to client
        await websocket.send_json({
            "event": "ready",
            "sample_rate": session.sample_rate,
            "session_id": session.session_id,
        })

        while is_running:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                session.add_chunk(message["bytes"])
            elif "text" in message and message["text"]:
                try:
                    cmd = json.loads(message["text"])
                    action = cmd.get("action", "")
                    if action == "start":
                        if "whisper_engine" in cmd:
                            session.whisper_engine = cmd["whisper_engine"]
                        if "whisper_model" in cmd:
                            session.whisper_model = cmd["whisper_model"]
                        if "language" in cmd:
                            session.language = cmd["language"]
                        if "vad_filter" in cmd:
                            session.vad_filter = bool(cmd["vad_filter"])
                        await websocket.send_json({
                            "event": "configured",
                            "engine": session.whisper_engine,
                            "model": session.whisper_model,
                            "language": session.language,
                        })
                    elif action == "stop":
                        break
                    elif action == "reset":
                        session.buffer.clear()
                        session.segments.clear()
                        session.current_partial_text = ""
                        await websocket.send_json({"event": "reset"})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", session.session_id)
    except Exception as e:
        logger.error("WebSocket streaming exception: %s", e)
        try:
            await websocket.send_json({"event": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        is_running = False
        proc_task.cancel()
        try:
            await proc_task
        except asyncio.CancelledError:
            pass

        # Finalize remaining audio
        try:
            async with processing_lock:
                final_result = await asyncio.to_thread(session.finish)
            await websocket.send_json({
                "event": "completed",
                "result": final_result.to_dict(),
            })
            await websocket.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Synchronous Audio Transcription Route (Backwards Compatibility)
# ------------------------------------------------------------------------------
@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    whisper_engine: str = Form(default="faster-whisper"),
    whisper_model: str = Form(default="base"),
    language: str = Form(default="auto"),
    vad_filter: bool = Form(default=True),
    enable_diarization: bool = Form(default=False),
    num_speakers: int = Form(default=-1),
):
    """Upload audio/video file, decode via in-memory PCM or WAV, and transcribe."""
    task_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".wav"
    temp_upload = UPLOADS_DIR / f"{task_id}_raw{ext}"

    start_time = time.time()
    try:
        with open(temp_upload, "wb") as f:
            shutil.copyfileobj(file.file, f)

        transcriber = transcriber_factory.get_transcriber(whisper_engine)

        from audio_processor import HAS_NUMPY
        use_memory = settings.use_in_memory_pcm and HAS_NUMPY and whisper_engine == "faster-whisper"

        if use_memory:
            audio_input, metadata = audio_processor.convert_to_pcm_array(temp_upload)
        else:
            converted_wav = UPLOADS_DIR / f"{task_id}_16k.wav"
            audio_input = audio_processor.convert_to_whisper_wav(temp_upload, converted_wav)
            metadata = audio_processor.probe_media(temp_upload)

        result = transcriber.transcribe(
            audio_input,
            model_name=whisper_model,
            language=language if language != "auto" else None,
            vad_filter=vad_filter,
        )

        num_speakers_detected = 0
        if enable_diarization:
            try:
                diarizer = diarizer_factory.get_diarizer("sherpa-onnx")
                if diarizer.is_available():
                    diar_res = diarizer.diarize(audio_input, num_speakers=num_speakers)
                    align_speakers_to_segments(result.segments, diar_res)
                    num_speakers_detected = diar_res.num_speakers
            except Exception as d_err:
                logger.warning("Synchronous diarization error: %s", d_err)

        elapsed = round(time.time() - start_time, 2)
        duration = metadata.duration if metadata else result.duration

        return {
            "task_id": task_id,
            "filename": file.filename,
            "duration": duration,
            "processing_time": elapsed,
            "engine_used": transcriber.name,
            "model_used": whisper_model,
            "language": result.language,
            "num_speakers": num_speakers_detected,
            "speaker_turns": result.get_speaker_turns(),
            "text": result.to_txt(),
            "srt": result.to_srt(),
            "vtt": result.to_vtt(),
            "ass": result.to_ass(),
            "word_vtt": result.to_word_vtt(),
            "segments": [s.model_dump() for s in result.segments],
        }

    except Exception as e:
        logger.exception("Transcription failed for task %s", task_id)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_upload.exists():
            try:
                temp_upload.unlink()
            except Exception:
                pass


# ------------------------------------------------------------------------------
# LLM Processing Route (Streaming SSE with Chunking Support)
# ------------------------------------------------------------------------------
@app.post("/api/process-llm")
async def process_llm(req: LLMProcessRequest):
    """Process transcript using Ollama or OpenAI-compatible provider with token streaming."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content is required.")

    try:
        prov = llm_registry.get_provider(req.provider)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {req.provider}")

    async def token_stream():
        try:
            if req.action == "polish":
                sys_prompt, prompt = get_polish_prompt(req.text)
                async for token in prov.generate_stream(
                    prompt=prompt, system_prompt=sys_prompt, model=req.model
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            else:
                summarizer = ChunkedSummarizer(prov)
                async for token in summarizer.summarize_stream(
                    transcript=req.text,
                    level=req.detail_level,
                    custom_instruction=req.custom_instruction,
                    model=req.model,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("LLM streaming generation error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------------------
# Notification Routes
# ------------------------------------------------------------------------------
@app.post("/api/notifications/test")
async def test_notification_provider(req: NotificationTestRequest):
    if req.provider == "telegram":
        notifier = TelegramNotifier(
            bot_token=req.telegram_bot_token or settings.telegram_bot_token,
            chat_id=req.telegram_chat_id or settings.telegram_chat_id,
            enabled=True,
        )
        res = await notifier.test_connection()
        return res.model_dump()

    elif req.provider == "webhook":
        notifier = WebhookNotifier(
            webhook_url=req.webhook_url or settings.webhook_url,
            secret=req.webhook_secret or settings.webhook_secret,
            enabled=True,
        )
        res = await notifier.test_connection()
        return res.model_dump()

    else:
        raise HTTPException(status_code=400, detail=f"Unknown notification provider: {req.provider}")


@app.post("/api/notify")
async def dispatch_notification(req: DispatchNotificationRequest):
    payload = NotificationPayload(
        task_id=req.task_id,
        filename=req.filename,
        duration_seconds=req.duration_seconds,
        processing_time_seconds=req.processing_time_seconds,
        whisper_model=req.whisper_model,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        summary_type=req.summary_type,
        transcript_text=req.transcript_text,
        summary_text=req.summary_text,
        polished_text=req.polished_text,
    )

    results = await dispatcher.dispatch_all(payload)
    return {"dispatched": len(results), "results": [r.model_dump() for r in results]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
