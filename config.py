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
_groq_cfg = _cfg.get("groq", {})
_openrouter_cfg = _cfg.get("openrouter", {})
_notif_cfg = _cfg.get("notifications", {})
_telegram_cfg = _notif_cfg.get("telegram", {})
_webhook_cfg = _notif_cfg.get("webhook", {})
_diarization_cfg = _cfg.get("diarization", {})
_audio_cfg = _cfg.get("audio", {})


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
    audio_enhancement_filter: str = Field(
        default_factory=lambda: _audio_cfg.get("enhancement_filter")
        or os.getenv(
            "AUDIO_ENHANCEMENT_FILTER",
            "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        )
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

    # Cloud STT & LLM - Groq
    groq_api_key: str | None = Field(
        default_factory=lambda: _groq_cfg.get("api_key") or os.getenv("GROQ_API_KEY")
    )
    groq_base_url: str = Field(
        default_factory=lambda: _groq_cfg.get("base_url") or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    groq_default_model: str = Field(
        default_factory=lambda: _groq_cfg.get("default_model") or os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
    )
    groq_default_stt_model: str = Field(
        default_factory=lambda: _groq_cfg.get("default_stt_model") or os.getenv("GROQ_DEFAULT_STT_MODEL", "whisper-large-v3-turbo")
    )

    # Cloud STT & LLM - OpenRouter
    openrouter_api_key: str | None = Field(
        default_factory=lambda: _openrouter_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: _openrouter_cfg.get("base_url") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_default_model: str = Field(
        default_factory=lambda: _openrouter_cfg.get("default_model") or os.getenv("OPENROUTER_DEFAULT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    )
    openrouter_default_stt_model: str = Field(
        default_factory=lambda: _openrouter_cfg.get("default_stt_model") or os.getenv("OPENROUTER_DEFAULT_STT_MODEL", "openai/whisper-large-v3")
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


def mask_secret(secret: str | None) -> str:
    """Mask secret API keys or tokens for secure UI display."""
    if not secret:
        return ""
    secret = str(secret).strip()
    if len(secret) <= 8:
        return "••••••••"
    return f"{secret[:4]}••••{secret[-4:]}"


def save_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Persist updated configuration to config.json and reload settings in-memory."""
    current_json = load_json_config(CONFIG_JSON_PATH)

    def deep_update(target: dict[str, Any], src: dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                deep_update(target[k], v)
            else:
                target[k] = v

    deep_update(current_json, updates)

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(current_json, f, indent=2)

    # Synchronize in-memory settings
    global settings
    for section_name, section_dict in updates.items():
        if not isinstance(section_dict, dict):
            continue
        for key, val in section_dict.items():
            # If secret wasn't modified (or was masked), don't overwrite with mask
            if isinstance(val, str) and "••••" in val:
                continue

            attr_name = f"{section_name}_{key}"
            if section_name == "whisper":
                attr_name = f"default_whisper_{key}" if key in ("engine", "model") else f"whisper_{key}"
            elif section_name == "server":
                attr_name = key
            elif section_name == "ollama":
                attr_name = f"default_ollama_model" if key == "default_model" else f"ollama_{key}"
            elif section_name == "openai_compatible":
                attr_name = f"openai_{key}"

            if hasattr(settings, attr_name):
                setattr(settings, attr_name, val)

    return get_masked_settings()


def get_masked_settings() -> dict[str, Any]:
    """Retrieve application settings with all sensitive API keys masked."""
    cfg = load_json_config(CONFIG_JSON_PATH)
    
    # Mask known secrets
    if "groq" in cfg and "api_key" in cfg["groq"]:
        cfg["groq"]["api_key"] = mask_secret(cfg["groq"]["api_key"])
    if "openrouter" in cfg and "api_key" in cfg["openrouter"]:
        cfg["openrouter"]["api_key"] = mask_secret(cfg["openrouter"]["api_key"])
    if "openai_compatible" in cfg and "api_key" in cfg["openai_compatible"]:
        cfg["openai_compatible"]["api_key"] = mask_secret(cfg["openai_compatible"]["api_key"])
    if "notifications" in cfg:
        notif = cfg["notifications"]
        if "telegram" in notif and "bot_token" in notif["telegram"]:
            notif["telegram"]["bot_token"] = mask_secret(notif["telegram"]["bot_token"])
        if "webhook" in notif and "secret" in notif["webhook"]:
            notif["webhook"]["secret"] = mask_secret(notif["webhook"]["secret"])

    # Fallback to runtime settings if config.json missing sections
    if "groq" not in cfg:
        cfg["groq"] = {
            "api_key": mask_secret(settings.groq_api_key),
            "base_url": settings.groq_base_url,
            "default_model": settings.groq_default_model,
            "default_stt_model": settings.groq_default_stt_model,
        }
    if "openrouter" not in cfg:
        cfg["openrouter"] = {
            "api_key": mask_secret(settings.openrouter_api_key),
            "base_url": settings.openrouter_base_url,
            "default_model": settings.openrouter_default_model,
            "default_stt_model": settings.openrouter_default_stt_model,
        }
    if "openai_compatible" not in cfg:
        cfg["openai_compatible"] = {
            "api_key": mask_secret(settings.openai_api_key),
            "base_url": settings.openai_base_url,
            "default_model": settings.openai_default_model,
        }

    return cfg
