"""Tests for configuration persistence and in-memory settings synchronization."""

from __future__ import annotations

import json
from pathlib import Path

import config
from config import mask_secret, save_config


def _write_config(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_save_config_does_not_overwrite_secret_with_mask(tmp_path, monkeypatch):
    """A masked secret echoed back by the UI must never replace the real secret."""
    cfg_file = tmp_path / "config.json"
    _write_config(cfg_file, {"groq": {"api_key": "REAL_SECRET_KEY_123456"}})
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "groq_api_key", "REAL_SECRET_KEY_123456")

    save_config({"groq": {"api_key": mask_secret("REAL_SECRET_KEY_123456")}})

    assert json.loads(cfg_file.read_text())["groq"]["api_key"] == "REAL_SECRET_KEY_123456"
    assert config.settings.groq_api_key == "REAL_SECRET_KEY_123456"


def test_save_config_persists_plain_secret(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    _write_config(cfg_file, {"groq": {"api_key": ""}})
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "groq_api_key", "")

    save_config({"groq": {"api_key": "brand_new_key_value"}})

    assert json.loads(cfg_file.read_text())["groq"]["api_key"] == "brand_new_key_value"
    assert config.settings.groq_api_key == "brand_new_key_value"


def test_save_config_applies_nested_notification_settings(tmp_path, monkeypatch):
    """Nested notifications.* updates must reach both config.json and runtime settings."""
    cfg_file = tmp_path / "config.json"
    _write_config(
        cfg_file,
        {"notifications": {"telegram": {"enabled": False, "bot_token": "", "chat_id": ""}}},
    )
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "")
    monkeypatch.setattr(config.settings, "telegram_enabled", False)

    save_config({"notifications": {"telegram": {"bot_token": "NEW_TOKEN", "chat_id": "999"}}})

    assert config.settings.telegram_bot_token == "NEW_TOKEN"
    assert config.settings.telegram_chat_id == "999"
    assert config.settings.telegram_enabled is True

    saved = json.loads(cfg_file.read_text())["notifications"]["telegram"]
    assert saved["bot_token"] == "NEW_TOKEN"
    assert saved["chat_id"] == "999"
    assert saved["enabled"] is True


def test_save_config_masked_notification_token_not_persisted(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    _write_config(
        cfg_file,
        {"notifications": {"telegram": {"enabled": True, "bot_token": "REAL_TOKEN_abcdefgh", "chat_id": "42"}}},
    )
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "REAL_TOKEN_abcdefgh")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "42")
    monkeypatch.setattr(config.settings, "telegram_enabled", True)

    save_config(
        {"notifications": {"telegram": {"bot_token": mask_secret("REAL_TOKEN_abcdefgh"), "chat_id": "42"}}}
    )

    saved = json.loads(cfg_file.read_text())["notifications"]["telegram"]
    assert saved["bot_token"] == "REAL_TOKEN_abcdefgh"
    assert config.settings.telegram_bot_token == "REAL_TOKEN_abcdefgh"


def test_save_config_disables_notification_when_credentials_cleared(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    _write_config(
        cfg_file,
        {"notifications": {"telegram": {"enabled": True, "bot_token": "TOK", "chat_id": "1"}}},
    )
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "TOK")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "1")
    monkeypatch.setattr(config.settings, "telegram_enabled", True)

    save_config({"notifications": {"telegram": {"bot_token": "", "chat_id": ""}}})

    assert config.settings.telegram_enabled is False
    assert json.loads(cfg_file.read_text())["notifications"]["telegram"]["enabled"] is False


def test_save_config_explicit_enabled_flag_wins_over_derivation(tmp_path, monkeypatch):
    """An explicit ``enabled`` value must not be clobbered by credential-based derivation."""
    cfg_file = tmp_path / "config.json"
    _write_config(
        cfg_file,
        {"notifications": {"telegram": {"enabled": True, "bot_token": "TOK", "chat_id": "1"}}},
    )
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "TOK")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "1")
    monkeypatch.setattr(config.settings, "telegram_enabled", True)

    save_config({"notifications": {"telegram": {"enabled": False}}})

    assert config.settings.telegram_enabled is False
    assert json.loads(cfg_file.read_text())["notifications"]["telegram"]["enabled"] is False


def test_dispatcher_reload_from_settings(monkeypatch):
    from notifications.dispatcher import NotificationDispatcher

    dispatcher = NotificationDispatcher(load_defaults=False)
    assert dispatcher.get_provider("telegram") is None

    monkeypatch.setattr(config.settings, "telegram_enabled", True)
    monkeypatch.setattr(config.settings, "telegram_bot_token", "bot_token_value")
    monkeypatch.setattr(config.settings, "telegram_chat_id", "12345")

    dispatcher.reload_from_settings()

    telegram = dispatcher.get_provider("telegram")
    assert telegram is not None
    assert telegram.is_configured() is True


def test_atomic_save_config(tmp_path, monkeypatch):
    """Verify save_config writes atomically and leaves no stray temporary files."""
    cfg_file = tmp_path / "config.json"
    _write_config(cfg_file, {"server": {"port": 8000}})
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", cfg_file)

    save_config({"server": {"port": 9000}})

    # Verify updated content
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["server"]["port"] == 9000

    # Verify no stray .tmp files left in directory
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0


def test_settings_host_default(monkeypatch):
    """Verify host defaults to 127.0.0.1 for local isolation unless configured otherwise."""
    monkeypatch.delenv("HOST", raising=False)
    from config import Settings
    s = Settings()
    # If no server.host in config, defaults to 127.0.0.1
    if not config._server_cfg.get("host"):
        assert s.host == "127.0.0.1"

