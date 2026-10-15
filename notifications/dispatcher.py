"""Notification Dispatcher module.

Manages all registered notification providers and coordinates async dispatch.
"""

import asyncio
import logging
from typing import Any

from config import settings
from notifications.base import BaseNotifier, NotificationPayload, NotificationResult
from notifications.telegram import TelegramNotifier
from notifications.webhook import WebhookNotifier

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    """Manages active notification providers and coordinates async dispatch."""

    def __init__(self, load_defaults: bool = True):
        self._providers: dict[str, BaseNotifier] = {}
        if load_defaults:
            self.reload_from_settings()

    def reload_from_settings(self) -> None:
        """(Re)build the default providers from the current runtime settings.

        Invoked after configuration changes so credential or enable/disable updates
        take effect immediately, without requiring a server restart.
        """
        self.register(
            TelegramNotifier(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
                enabled=settings.telegram_enabled,
            )
        )
        self.register(
            WebhookNotifier(
                webhook_url=settings.webhook_url,
                secret=settings.webhook_secret,
                enabled=settings.webhook_enabled,
            )
        )

    def register(self, provider: BaseNotifier) -> None:
        """Register a notification provider."""
        self._providers[provider.name] = provider
        logger.info("Registered notification provider: %s (%s)", provider.name, provider.display_name)

    def get_provider(self, name: str) -> BaseNotifier | None:
        """Retrieve a specific provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[dict[str, Any]]:
        """List all registered providers and their configured statuses."""
        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "configured": p.is_configured(),
            }
            for p in self._providers.values()
        ]

    async def dispatch_all(self, payload: NotificationPayload) -> list[NotificationResult]:
        """Broadcast payload to all configured providers concurrently."""
        active_providers = [p for p in self._providers.values() if p.is_configured()]
        if not active_providers:
            logger.debug("No configured notification providers found for dispatch.")
            return []

        tasks = [provider.send(payload) for provider in active_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results: list[NotificationResult] = []
        for provider, res in zip(active_providers, results):
            if isinstance(res, Exception):
                logger.error("Provider %s failed with uncaught exception: %s", provider.name, res)
                processed_results.append(
                    NotificationResult(
                        provider=provider.name,
                        success=False,
                        message="Internal dispatch exception",
                        error_details=str(res),
                    )
                )
            else:
                processed_results.append(res)

        return processed_results


# Global notification dispatcher instance
dispatcher = NotificationDispatcher()
