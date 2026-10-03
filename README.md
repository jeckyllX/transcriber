# Audio Transcriber

Audio and video transcription tool powered by Whisper, ONNX speaker diarization, and optional LLM post-processing. Converts media to 16kHz mono WAV via FFmpeg, transcribes using local or API Whisper engines, and optionally polishes or summarizes transcripts using local or cloud LLMs.

<p align="center">
  <img src="docs/screenshots/dashboard.png" alt="Audio Transcriber Dashboard" width="100%">
</p>

---

## Features

- **Audio Ingestion & Conversion**: Ingests `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.opus`, `.webm`, `.mp4`, `.mkv`, etc., and standardizes to 16,000 Hz, 1-channel mono, 16-bit PCM WAV via FFmpeg.
- **Whisper Transcription Engine**:
  - `faster-whisper` (CTranslate2 with CPU `int8` quantization).
  - `whisper.cpp` standalone C++ binary runner with SHA256 verification and compilation fallback.
  - Cloud STT via OpenAI-compatible endpoints (Groq Whisper, OpenRouter).
  - Subtitle and transcript export: plain text, timestamped segments, **SRT**, **WebVTT**, **ASS**, **JSON**.
- **Speaker Diarization (ONNX)**:
  - Offline speaker identification and turn segmentation using `sherpa-onnx` (Pyannote 3.0 segmentation + 3D-Speaker embedding models on CPU).
  - On-demand model download on first use.
  - Inline speaker renaming that updates exports and dialogue turns.
- **Live Dictation (WebSockets)**:
  - Real-time microphone audio streaming over WebSockets (`/api/ws/transcribe`).
  - Sliding-window decoding with energy-based VAD and silence detection.
  - Interactive live UI with transcript feed and export handoff.
- **Word-Level Timestamps & Playback**:
  - Word-level timestamps and confidence scores.
  - Audio player with click-to-seek navigation (clicking any word seeks audio playback to that position).
  - Playback highlighting synchronized with audio.
- **LLM Post-Processing**:
  - Transcript polishing (grammar and punctuation cleanup) and summarization (TL;DR, key takeaways, detailed, action items).
  - Supports local Ollama or any OpenAI-compatible API (Groq, OpenRouter, self-hosted vLLM/llama.cpp, etc.).
  - Streaming token output.
- **Notifications**:
  - Telegram bot: sends execution stats, formatted summaries, and transcript files (`.txt` / `.md`).
  - Webhook: dispatches JSON POST payloads on job completion.
- **Web UI**:
  - Single-page interface with drag-and-drop file upload and microphone recording.
  - Transcript view with speaker dialogue grouping and export options.
  - Settings modal for configuring API keys, endpoints, and notifications.
  - Zero Node/NPM dependencies needed to run.

---

## Quick Start

### 1. Automated Installation
The portable installer detects your system architecture (`x86_64` or `arm64`), checks FFmpeg, downloads Whisper models with SHA256 verification, and configures the Python virtual environment:

```bash
bash install.sh
```

### 2. Launch Application
Start the server on `http://localhost:8000`:

```bash
bash run.sh
```

---

## Configuration (`config.json`)

Copy `config.json.example` to `config.json` to customize your settings:

```bash
cp config.json.example config.json
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server HTTP port |
| `DEFAULT_WHISPER_MODEL`| `base` | Default Whisper model (`tiny`, `base`, `small`, `medium`) |
| `GROQ_API_KEY` | | Groq API Key |
| `OPENROUTER_API_KEY` | | OpenRouter API Key |
| `OPENAI_API_KEY` | | OpenAI or OpenAI-compatible API Key |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama API endpoint |
| `DEFAULT_OLLAMA_MODEL` | `llama3.2` | Default model for polishing and summaries |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram notification dispatch |
| `TELEGRAM_BOT_TOKEN` | | Telegram Bot Token from `@BotFather` |
| `TELEGRAM_CHAT_ID` | | Target Chat or Channel ID |
| `WEBHOOK_ENABLED` | `false` | Enable generic Webhook dispatch |
| `WEBHOOK_URL` | | HTTP POST endpoint for notifications |

---

## Telegram Setup Guide

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot` and follow the instructions to get your **Bot Token** (e.g. `123456789:ABCdefGhIJKlmNoPQRstuVWXyz`).
3. Start a chat with your new bot and send `/start`.
4. Get your **Chat ID** by messaging `@userinfobot` or checking `https://api.telegram.org/bot<TOKEN>/getUpdates`.
5. Enter the Token and Chat ID in the Web UI **Settings** modal and click **Test Connection**.

---

## Running Automated Tests

Run the full regression test suite:

```bash
.venv/bin/pytest tests/ -v
```

---

## Architecture Overview

```
transcriber/
├── audio_processor.py         # FFmpeg universal conversion & media metadata probe
├── config.py                  # Zero-hardcoding, dynamic environment configuration
├── jobs.py                    # Job lifecycle management, semaphore concurrency, and SSE
├── main.py                    # FastAPI server & Server-Sent Events (SSE) streaming
├── install.sh                 # Architecture-aware installer with SHA256 validation
├── run.sh                     # Application runner
├── diarization/               # Lightweight Offline Diarization Subsystem (Sherpa-ONNX)
│   ├── base.py                # BaseDiarizer, SpeakerInterval & DiarizationResult DTOs
│   ├── sherpa_diarizer.py     # Pyannote 3.0 + 3D-Speaker ONNX runner (on-demand download)
│   ├── alignment.py           # Temporal overlap alignment mapping speakers to segments/words
│   └── factory.py             # Diarizer dynamic factory and engine registry
├── notifications/             # Modular Notification System (Open/Closed Principle)
│   ├── base.py                # BaseNotifier & NotificationPayload DTO
│   ├── telegram.py            # TelegramNotifier (auto-chunking & document attachments)
│   ├── webhook.py             # Generic WebhookNotifier
│   └── dispatcher.py          # Asynchronous concurrent dispatcher
├── transcribers/              # Pluggable Transcriber Subsystem
│   ├── base.py                # BaseTranscriber, TranscriptionResult (TXT, SRT, VTT, ASS, JSON)
│   ├── faster_whisper.py      # CTranslate2 Python engine
│   ├── whisper_cpp.py         # Standalone C++ binary adapter
│   ├── openai_compat.py       # Universal cloud STT adapter (Groq, OpenRouter, OpenAI, vLLM)
│   ├── streaming.py           # Real-time WebSocket audio ring buffer & live session
│   └── factory.py             # Dynamic transcriber resolution
├── llm/                       # Modular AI / LLM Subsystem
│   ├── base.py                # BaseLLMProvider interface
│   ├── ollama.py              # Ollama client with token streaming & model pulling
│   ├── openai_compat.py       # Universal OpenAI-compatible client (Groq, OpenRouter, vLLM)
│   ├── prompts.py             # Decoupled prompts (Polish + 5 summary tiers with speaker headers)
│   └── registry.py            # LLM provider registry
├── templates/
│   └── index.html             # Web UI template
├── static/
│   ├── app.js                 # UI controller, mic recording, SSE streams, live dictation
│   └── style.css              # Custom CSS
└── tests/                     # 50 automated tests covering all subsystems
```

---

## License

This project is licensed under the [MIT License](LICENSE).
