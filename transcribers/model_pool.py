"""Thread-safe generic model pool with active lease tracking and idle eviction.

Provides concurrency control and memory management for heavy machine learning
models, ensuring models actively in use are never evicted during background sweeps.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import logging
import threading
import time
from typing import Any, Callable, Generic, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ModelEntry(Generic[T]):
    """Internal container tracking a cached model instance and its usage lifecycle."""

    name: str
    instance: T
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    active_leases: int = 0


class ManagedModelPool(Generic[T]):
    """Thread-safe pool managing cached model instances with lease tracking.

    Guarantees:
    1. Active usage tracking: models leased via `lease()` increment active_leases.
    2. Eviction safety: models with `active_leases > 0` are NEVER evicted or unloaded.
    3. Idle expiration: models unused for >= `timeout_seconds` are safely evicted.
    4. Thread-safe synchronization across all load, lease, and eviction operations.
    """

    def __init__(
        self,
        loader: Callable[[str], T],
        disposer: Callable[[T], None] | None = None,
        name: str = "model-pool",
    ):
        self.loader = loader
        self.disposer = disposer
        self.name = name
        self._models: dict[str, ModelEntry[T]] = {}
        self._lock = threading.RLock()

    def _normalize_name(self, model_name: str) -> str:
        return model_name.replace("ggml-", "").replace(".bin", "").strip()

    def get_or_load(self, model_name: str) -> T:
        """Retrieve existing cached model or instantiate a new one under lock.

        Updates last_used_at timestamp.
        """
        clean_name = self._normalize_name(model_name)
        with self._lock:
            if clean_name not in self._models:
                logger.info(
                    "[%s] Loading model '%s' into memory...",
                    self.name,
                    clean_name,
                )
                instance = self.loader(clean_name)
                self._models[clean_name] = ModelEntry(
                    name=clean_name,
                    instance=instance,
                    created_at=time.time(),
                    last_used_at=time.time(),
                    active_leases=0,
                )
            else:
                self._models[clean_name].last_used_at = time.time()

            return self._models[clean_name].instance

    @contextmanager
    def lease(self, model_name: str) -> Iterator[T]:
        """Lease a model instance within a managed context.

        Increments active_leases upon entry and decrements upon exit (guaranteed
        via try/finally). Refreshes last_used_at timestamp upon release.
        While leased, this model cannot be evicted by any idle cleanup sweep.
        """
        clean_name = self._normalize_name(model_name)
        with self._lock:
            if clean_name not in self._models:
                logger.info(
                    "[%s] Loading model '%s' into memory...",
                    self.name,
                    clean_name,
                )
                instance = self.loader(clean_name)
                self._models[clean_name] = ModelEntry(
                    name=clean_name,
                    instance=instance,
                    created_at=time.time(),
                    last_used_at=time.time(),
                    active_leases=0,
                )

            entry = self._models[clean_name]
            entry.active_leases += 1
            entry.last_used_at = time.time()
            instance = entry.instance

        try:
            yield instance
        finally:
            with self._lock:
                entry.active_leases = max(0, entry.active_leases - 1)
                entry.last_used_at = time.time()

    def evict_idle(self, timeout_seconds: float) -> list[str]:
        """Evict models that have been idle for >= timeout_seconds.

        Models with active_leases > 0 are strictly preserved and never evicted.
        Returns a list of evicted model names.
        """
        if timeout_seconds <= 0:
            return []

        now = time.time()
        evicted: list[str] = []
        with self._lock:
            to_remove: list[tuple[str, float]] = []
            for name, entry in self._models.items():
                if entry.active_leases > 0:
                    logger.debug(
                        "[%s] Model '%s' is actively processing (%d active tasks); skipping eviction.",
                        self.name,
                        name,
                        entry.active_leases,
                    )
                    continue

                idle_time = now - entry.last_used_at
                if idle_time >= timeout_seconds:
                    to_remove.append((name, idle_time))

            for name, idle_time in to_remove:
                entry = self._models.pop(name)
                if self.disposer:
                    try:
                        self.disposer(entry.instance)
                    except Exception as e:
                        logger.warning("[%s] Error disposing model '%s': %s", self.name, name, e)
                del entry.instance
                del entry
                evicted.append(name)
                logger.info(
                    "[%s] Evicted idle model '%s' (idle for %.1fs >= %ds threshold).",
                    self.name,
                    name,
                    idle_time,
                    timeout_seconds,
                )

        if evicted:
            gc.collect()

        return evicted

    def unload_idle(self, model_name: str | None = None) -> list[str]:
        """Unload specified or all cached models that are not actively in use.

        Returns list of model names that were successfully unloaded.
        """
        evicted: list[str] = []
        with self._lock:
            if model_name:
                clean_name = self._normalize_name(model_name)
                entry = self._models.get(clean_name)
                if entry and entry.active_leases == 0:
                    self._models.pop(clean_name)
                    if self.disposer:
                        try:
                            self.disposer(entry.instance)
                        except Exception as e:
                            logger.warning("[%s] Error disposing model '%s': %s", self.name, clean_name, e)
                    del entry.instance
                    del entry
                    evicted.append(clean_name)
                    logger.info("[%s] Manually unloaded idle model '%s'.", self.name, clean_name)
            else:
                to_remove = [k for k, v in self._models.items() if v.active_leases == 0]
                for name in to_remove:
                    entry = self._models.pop(name)
                    if self.disposer:
                        try:
                            self.disposer(entry.instance)
                        except Exception as e:
                            logger.warning("[%s] Error disposing model '%s': %s", self.name, name, e)
                    del entry.instance
                    del entry
                    evicted.append(name)
                    logger.info("[%s] Manually unloaded idle model '%s'.", self.name, name)

        if evicted:
            gc.collect()

        return evicted

    def is_in_use(self, model_name: str) -> bool:
        """Check if a specific model is currently leased and actively processing."""
        clean_name = self._normalize_name(model_name)
        with self._lock:
            entry = self._models.get(clean_name)
            return bool(entry and entry.active_leases > 0)

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of cached models, usage state, and idle durations."""
        now = time.time()
        with self._lock:
            return {
                name: {
                    "in_use": entry.active_leases > 0,
                    "active_leases": entry.active_leases,
                    "idle_seconds": round(now - entry.last_used_at, 2) if entry.active_leases == 0 else 0.0,
                    "last_used_at": entry.last_used_at,
                    "created_at": entry.created_at,
                }
                for name, entry in self._models.items()
            }

    def get_active_instances(self) -> dict[str, T]:
        """Return a dictionary mapping model names to their loaded instances."""
        with self._lock:
            return {name: entry.instance for name, entry in self._models.items()}


class AsyncModelEvictionMonitor:
    """Periodic asynchronous task that triggers idle eviction sweeps on registered pools or factories."""

    def __init__(
        self,
        sweep_fn: Callable[[float], dict[str, list[str]]],
        check_interval_seconds: float = 60.0,
    ):
        self.sweep_fn = sweep_fn
        self.check_interval_seconds = check_interval_seconds
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def _loop(self, timeout_seconds: float) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                evicted = self.sweep_fn(timeout_seconds)
                for engine_id, models in evicted.items():
                    if models:
                        logger.info("Evicted idle models from engine '%s': %s", engine_id, ", ".join(models))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error in model eviction monitor: %s", e)

    def start(self, timeout_seconds: float) -> None:
        """Start periodic background eviction sweep."""
        if self._running or timeout_seconds <= 0:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(timeout_seconds))
        logger.info(
            "Model idle eviction monitor started (timeout=%ds, check_interval=%.0fs).",
            timeout_seconds,
            self.check_interval_seconds,
        )

    async def stop(self) -> None:
        """Gracefully stop periodic background eviction sweep."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Model idle eviction monitor stopped.")

