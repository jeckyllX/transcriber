"""Factory for creating and managing speaker diarization engines."""

from __future__ import annotations

from typing import Type
from diarization.base import BaseDiarizer
from diarization.sherpa_diarizer import SherpaDiarizer


class DiarizerFactory:
    """Registry and factory for speaker diarization engines."""

    def __init__(self):
        self._engines: dict[str, Type[BaseDiarizer]] = {}
        self._instances: dict[str, BaseDiarizer] = {}
        self.register(SherpaDiarizer.name, SherpaDiarizer)

    def register(self, name: str, diarizer_cls: Type[BaseDiarizer]) -> None:
        self._engines[name.lower()] = diarizer_cls

    def get_diarizer(self, name: str = "sherpa-onnx") -> BaseDiarizer:
        name_lower = name.lower()
        if name_lower not in self._instances:
            if name_lower not in self._engines:
                raise KeyError(f"Unknown diarization engine: '{name}'. Available: {list(self._engines.keys())}")
            self._instances[name_lower] = self._engines[name_lower]()
        return self._instances[name_lower]

    def list_engines(self) -> list[dict[str, str | bool]]:
        res = []
        for name, cls in self._engines.items():
            instance = self.get_diarizer(name)
            res.append({
                "name": name,
                "display_name": instance.display_name,
                "available": instance.is_available(),
                "models_ready": getattr(instance, "models_ready", lambda: True)(),
            })
        return res


diarizer_factory = DiarizerFactory()
