"""Sherpa-ONNX implementation of CPU-only Speaker Diarization."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any
import numpy as np

from config import settings
from diarization.base import BaseDiarizer, DiarizationResult, SpeakerInterval

logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    sherpa_onnx = None
    HAS_SHERPA = False


class SherpaDiarizer(BaseDiarizer):
    """Offline speaker diarization using Pyannote segmentation and 3D-Speaker/WeSpeaker via Sherpa-ONNX."""

    name: str = "sherpa-onnx"
    display_name: str = "Sherpa ONNX Diarizer"

    def __init__(
        self,
        seg_model_path: Path | str | None = None,
        emb_model_path: Path | str | None = None,
    ):
        self._custom_seg_model = Path(seg_model_path) if seg_model_path else None
        self._custom_emb_model = Path(emb_model_path) if emb_model_path else None

    def _resolve_model_path(self, configured_path: Path | str | None, default_filename: str) -> Path:
        """Resolve model path relative to models directory or absolute path."""
        if configured_path:
            p = Path(configured_path)
            if p.is_file():
                return p
            p_in_models = settings.whisper_models_dir / p
            if p_in_models.is_file():
                return p_in_models

        # Fallback to default in models directory
        candidate = settings.whisper_models_dir / default_filename
        return candidate

    SEG_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    EMB_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    @property
    def segmentation_model_path(self) -> Path:
        p = self._resolve_model_path(
            self._custom_seg_model or getattr(settings, "diarization_seg_model", None),
            "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        )
        # Check int8 quantized alternative if standard model is missing
        if not p.is_file():
            p_int8 = p.parent / "model.int8.onnx"
            if p_int8.is_file():
                return p_int8
        return p

    @property
    def embedding_model_path(self) -> Path:
        return self._resolve_model_path(
            self._custom_emb_model or getattr(settings, "diarization_emb_model", None),
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        )

    def models_ready(self) -> bool:
        """Check if ONNX model files are already downloaded locally."""
        return self.segmentation_model_path.is_file() and self.embedding_model_path.is_file()

    def is_available(self) -> bool:
        """Verify Sherpa-ONNX library is installed and capable of running."""
        return HAS_SHERPA and sherpa_onnx is not None

    def ensure_models(self, status_callback: Any = None) -> bool:
        """Download missing ONNX models on demand into settings.whisper_models_dir."""
        if self.models_ready():
            return True

        import urllib.request
        import tarfile

        models_dir = settings.whisper_models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        if not self.segmentation_model_path.is_file():
            if status_callback:
                status_callback("Downloading speaker segmentation model (6 MB)...")
            logger.info("Downloading Sherpa-ONNX segmentation model on demand from %s", self.SEG_MODEL_URL)
            tar_path = models_dir / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
            urllib.request.urlretrieve(self.SEG_MODEL_URL, tar_path)
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=models_dir)
            if tar_path.is_file():
                tar_path.unlink()
            logger.info("Segmentation model extracted to %s", models_dir)

        if not self.embedding_model_path.is_file():
            if status_callback:
                status_callback("Downloading speaker embedding model (37 MB)...")
            logger.info("Downloading Sherpa-ONNX embedding model on demand from %s", self.EMB_MODEL_URL)
            temp_emb = models_dir / "embedding_download.tmp"
            urllib.request.urlretrieve(self.EMB_MODEL_URL, temp_emb)
            temp_emb.rename(self.embedding_model_path)
            logger.info("Embedding model saved to %s", self.embedding_model_path)

        return self.models_ready()

    def _load_audio_samples(self, audio: Path | Any) -> np.ndarray:
        """Load audio into a 1-D float32 numpy array sampled at 16000 Hz."""
        if isinstance(audio, np.ndarray):
            arr = audio
            if arr.ndim > 1:
                arr = arr[:, 0]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            # Normalize if 16-bit integer range
            if np.max(np.abs(arr)) > 1.0:
                arr = arr / 32768.0
            return np.ascontiguousarray(arr)

        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If it's a WAV file with 16kHz mono, read directly with wave
        try:
            with wave.open(str(audio_path), "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                if channels == 1 and rate == 16000 and sampwidth == 2:
                    frames = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    return samples
        except Exception:
            pass

        # Fallback to AudioProcessor for decoding any format to 16kHz PCM
        from audio_processor import AudioProcessor
        processor = AudioProcessor()
        samples, _ = processor.convert_to_pcm_array(audio_path)
        return samples

    def diarize(
        self,
        audio: Path | Any,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        **kwargs: Any,
    ) -> DiarizationResult:
        """Run offline speaker diarization and return temporal intervals."""
        if not self.is_available():
            raise RuntimeError("SherpaDiarizer is not available. Please install 'sherpa-onnx'.")

        if not self.models_ready():
            logger.info("Diarization models not found locally; downloading on demand...")
            self.ensure_models()

        samples = self._load_audio_samples(audio)
        duration_sec = len(samples) / 16000.0

        # Minimum required speech duration for meaningful diarization
        if duration_sec < 0.5:
            logger.info("Audio duration (%.2fs) too short for diarization; returning empty result", duration_sec)
            return DiarizationResult(num_speakers=0, intervals=[])

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model_path)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model_path)
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers,
                threshold=cluster_threshold,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )

        if not config.validate():
            raise RuntimeError(f"Sherpa-ONNX Diarization config validation failed: {config}")

        sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        raw_result = sd.process(samples)

        intervals: list[SpeakerInterval] = []
        for segment in raw_result.sort_by_start_time():
            intervals.append(
                SpeakerInterval(
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    speaker=f"Speaker {segment.speaker}",
                    confidence=round(float(segment.confidence), 3) if segment.confidence is not None else None,
                )
            )

        num_detected = raw_result.num_speakers if raw_result.num_speakers > 0 else len({i.speaker for i in intervals})

        return DiarizationResult(
            num_speakers=num_detected,
            intervals=intervals,
        )
