"""Application configuration module.

Loads settings from a unified modern JSON configuration file (`config.json`),
with automatic discovery of system binaries and dynamic path resolution.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

# Base directories dynamically anchored to project root
BASE_DIR: Path = Path(__file__).resolve().parent
BIN_DIR: Path = BASE_DIR / "bin"
MODELS_DIR: Path = BASE_DIR / "models"
UPLOADS_DIR: Path = BASE_DIR / "uploads"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"
CONFIG_JSON_PATH: Path = BASE_DIR / "config.json"

# Ensure runtime directories exist
for directory in (BIN_DIR, MODELS_DIR, UPLOADS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def find_binary(binary_name: str, fallback_dir: Path = BIN_DIR) -> str | None:
    """Find binary in system PATH first, then fallback to local bin directory."""
    system_path = shutil.which(binary_name)
    if system_path:
        return system_path
    local_path = fallback_dir / binary_name
    if local_path.is_file() and os.access(local_path, os.X_OK):
        return str(local_path)
    return None


def load_json_config(path: Path) -> dict[str, Any]:
    """Load configuration from a JSON file, gracefully stripping single-line comments if any."""
    if not path.is_file():
        return {}
    try:
        raw_text = path.read_text(encoding="utf-8")
        # Strip potential // comments for user convenience
        cleaned_text = re.sub(r"^\s*//.*$", "", raw_text, flags=re.MULTILINE)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"[WARN] Failed to parse {path.name}: {e}. Falling back to default settings.")
        return {}


# Load raw JSON dictionary if config.json exists
_cfg = load_json_config(CONFIG_JSON_PATH)

_server_cfg = _cfg.get("server", {})
_whisper_cfg = _cfg.get("whisper", {})
_ollama_cfg = _cfg.get("ollama", {})
_openai_cfg = _cfg.get("openai_compatible", {})
_notif_cfg = _cfg.get("notifications", {})
_telegram_cfg = _notif_cfg.get("telegram", {})
_webhook_cfg = _notif_cfg.get("webhook", {})
_diarization_cfg = _cfg.get("diarization", {})


class Settings(BaseModel):
    """Application settings loaded from config.json with fallback to environment."""

    # Server Settings
    host: str = Field(
        default_factory=lambda: _server_cfg.get("host") or os.getenv("HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(_server_cfg.get("port") or os.getenv("PORT", "8000"))
    )
    debug: bool = Field(
        default_factory=lambda: bool(_server_cfg.get("debug", False)) or os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
    )
    max_concurrent_jobs: int = Field(
        default_factory=lambda: int(_server_cfg.get("max_concurrent_jobs") or os.getenv("MAX_CONCURRENT_JOBS", "1"))
    )

    # Whisper Settings
    default_whisper_engine: str = Field(
        default_factory=lambda: _whisper_cfg.get("default_engine") or os.getenv("DEFAULT_WHISPER_ENGINE", "faster-whisper")
    )
    default_whisper_model: str = Field(
        default_factory=lambda: _whisper_cfg.get("default_model") or os.getenv("DEFAULT_WHISPER_MODEL", "base")
    )
    whisper_bin: str | None = Field(
        default_factory=lambda: _whisper_cfg.get("bin_path") or os.getenv("WHISPER_BIN_PATH") or find_binary("whisper-cli") or find_binary("main")
    )
    use_in_memory_pcm: bool = Field(
        default_factory=lambda: bool(_whisper_cfg.get("use_in_memory_pcm", True))
    )
    whisper_models_dir: Path = Field(
        default_factory=lambda: Path(_whisper_cfg.get("models_dir") or os.getenv("WHISPER_MODELS_DIR", str(MODELS_DIR)))
    )

    # Audio Processor Settings
    ffmpeg_bin: str | None = Field(
        default_factory=lambda: os.getenv("FFMPEG_BIN_PATH") or find_binary("ffmpeg")
    )
    ffprobe_bin: str | None = Field(
        default_factory=lambda: os.getenv("FFPROBE_BIN_PATH") or find_binary("ffprobe")
    )

    # LLM Provider - Ollama
    ollama_base_url: str = Field(
        default_factory=lambda: _ollama_cfg.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    default_ollama_model: str = Field(
        default_factory=lambda: _ollama_cfg.get("default_model") or os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2")
    )

    # LLM Provider - OpenAI-Compatible (Optional)
    openai_api_key: str | None = Field(
        default_factory=lambda: _openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str = Field(
        default_factory=lambda: _openai_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_default_model: str = Field(
        default_factory=lambda: _openai_cfg.get("default_model") or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    )

    # Notifications - Telegram
    telegram_enabled: bool = Field(
        default_factory=lambda: bool(_telegram_cfg.get("enabled", False)) or os.getenv("TELEGRAM_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    telegram_bot_token: str | None = Field(
        default_factory=lambda: _telegram_cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: str | None = Field(
        default_factory=lambda: _telegram_cfg.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID")
    )

    # Notifications - Generic Webhook
    webhook_enabled: bool = Field(
        default_factory=lambda: bool(_webhook_cfg.get("enabled", False)) or os.getenv("WEBHOOK_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    webhook_url: str | None = Field(
        default_factory=lambda: _webhook_cfg.get("url") or os.getenv("WEBHOOK_URL")
    )
    webhook_secret: str | None = Field(
        default_factory=lambda: _webhook_cfg.get("secret") or os.getenv("WEBHOOK_SECRET")
    )

    # Speaker Diarization Settings (Sherpa-ONNX)
    diarization_enabled: bool = Field(
        default_factory=lambda: bool(_diarization_cfg.get("enabled", True)) or os.getenv("DIARIZATION_ENABLED", "true").lower() in ("1", "true", "yes")
    )
    diarization_seg_model: str = Field(
        default_factory=lambda: _diarization_cfg.get("seg_model") or os.getenv("DIARIZATION_SEG_MODEL", "sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
    )
    diarization_emb_model: str = Field(
        default_factory=lambda: _diarization_cfg.get("emb_model") or os.getenv("DIARIZATION_EMB_MODEL", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
    )
    diarization_threshold: float = Field(
        default_factory=lambda: float(_diarization_cfg.get("threshold", 0.5) if _diarization_cfg.get("threshold") is not None else os.getenv("DIARIZATION_THRESHOLD", "0.5"))
    )


# Global singleton settings instance
settings = Settings()
