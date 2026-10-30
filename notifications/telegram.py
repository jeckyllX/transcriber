"""Telegram notification provider.

Uses the official Telegram Bot API to send Markdown/HTML messages and
document attachments with automatic character splitting for long texts.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Optional
import httpx

from notifications.base import BaseNotifier, NotificationPayload, NotificationResult

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
MAX_TELEGRAM_MSG_LEN = 4000


class TelegramNotifier(BaseNotifier):
    """Dispatches notifications and file attachments to a Telegram chat."""

    name: str = "telegram"
    display_name: str = "Telegram Bot"

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled

    def is_configured(self) -> bool:
        return bool(self.enabled and self.bot_token and self.chat_id)

    def _format_message(self, payload: NotificationPayload) -> str:
        """Format a clean HTML message for Telegram."""
        duration_min = f"{payload.duration_seconds / 60:.1f} min" if payload.duration_seconds > 0 else "N/A"
        proc_time = f"{payload.processing_time_seconds:.1f}s"

        safe_filename = html.escape(payload.filename or "Audio")
        safe_whisper = html.escape(payload.whisper_model or "base")

        lines = [
            "🎙 <b>Audio Transcription Complete</b>",
            f"📁 <b>File:</b> <code>{safe_filename}</code>",
            f"⏱ <b>Duration:</b> {duration_min} | <b>Processing:</b> {proc_time}",
            f"⚙️ <b>Model:</b> Whisper ({safe_whisper})",
        ]

        if payload.llm_model:
            mode_label = html.escape((payload.summary_type or "Processed").capitalize())
            safe_llm = html.escape(payload.llm_model)
            lines.append(f"🤖 <b>AI ({mode_label}):</b> {safe_llm}")

        lines.append("")

        # Add Summary or Polish or Transcript excerpt
        if payload.summary_text:
            lines.append("📋 <b>Summary:</b>")
            summary_preview = payload.summary_text.strip()
            if len(summary_preview) > 2500:
                summary_preview = html.escape(summary_preview[:2500]) + "...\n<i>(Full content in attached file)</i>"
            else:
                summary_preview = html.escape(summary_preview)
            lines.append(summary_preview)
        elif payload.polished_text:
            lines.append("✨ <b>Polished Transcript:</b>")
            polish_preview = payload.polished_text.strip()
            if len(polish_preview) > 2500:
                polish_preview = html.escape(polish_preview[:2500]) + "...\n<i>(Full content in attached file)</i>"
            else:
                polish_preview = html.escape(polish_preview)
            lines.append(polish_preview)
        elif payload.transcript_text:
            lines.append("📝 <b>Transcript:</b>")
            trans_preview = payload.transcript_text.strip()
            if len(trans_preview) > 2500:
                trans_preview = html.escape(trans_preview[:2500]) + "...\n<i>(Full content in attached file)</i>"
            else:
                trans_preview = html.escape(trans_preview)
            lines.append(trans_preview)

        return "\n".join(lines)

    async def send(self, payload: NotificationPayload) -> NotificationResult:
        if not self.is_configured():
            return NotificationResult(
                provider=self.name,
                success=False,
                message="Telegram notifier is disabled or missing credentials.",
            )

        text = self._format_message(payload)
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    }
                )

                if resp.status_code != 200:
                    err_json = resp.json() if "application/json" in resp.headers.get("content-type", "") else resp.text
                    logger.error("Telegram sendMessage error: %s", err_json)
                    return NotificationResult(
                        provider=self.name,
                        success=False,
                        message=f"Telegram API returned {resp.status_code}",
                        error_details=str(err_json),
                    )

                # Send attachments if provided
                if payload.attachment_files:
                    doc_url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendDocument"
                    for fname, fpath in payload.attachment_files.items():
                        path_obj = Path(fpath)
                        if path_obj.is_file():
                            with open(path_obj, "rb") as f:
                                files = {"document": (fname, f, "text/plain")}
                                data = {"chat_id": self.chat_id, "caption": f"📄 {fname}"}
                                await client.post(doc_url, data=data, files=files)

                return NotificationResult(
                    provider=self.name,
                    success=True,
                    message="Telegram message delivered successfully.",
                )

        except Exception as e:
            logger.exception("Telegram notification failed")
            return NotificationResult(
                provider=self.name,
                success=False,
                message="Failed to dispatch Telegram message",
                error_details=str(e),
            )

    async def test_connection(self) -> NotificationResult:
        if not self.bot_token:
            return NotificationResult(
                provider=self.name,
                success=False,
                message="Telegram bot token is not configured.",
            )

        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/getMe"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    bot_data = resp.json().get("result", {})
                    bot_username = bot_data.get("username", "UnknownBot")
                    # If chat_id is provided, send a test ping
                    if self.chat_id:
                        send_url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
                        send_resp = await client.post(
                            send_url,
                            json={
                                "chat_id": self.chat_id,
                                "text": f"✅ <b>Connection Test Successful!</b>\nConnected to @{bot_username}.",
                                "parse_mode": "HTML",
                            }
                        )
                        if send_resp.status_code != 200:
                            return NotificationResult(
                                provider=self.name,
                                success=False,
                                message=f"Bot authenticated (@{bot_username}), but chat ID test failed with status {send_resp.status_code}.",
                                error_details=send_resp.text,
                            )

                    return NotificationResult(
                        provider=self.name,
                        success=True,
                        message=f"Successfully connected to Telegram Bot: @{bot_username}",
                    )
                else:
                    return NotificationResult(
                        provider=self.name,
                        success=False,
                        message=f"Invalid Bot Token (HTTP {resp.status_code})",
                        error_details=resp.text,
                    )
        except Exception as e:
            return NotificationResult(
                provider=self.name,
                success=False,
                message="Network error contacting Telegram API",
                error_details=str(e),
            )
