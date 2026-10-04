"""Audio processing module.

Handles universal media conversion to Whisper's standardized audio format:
16,000 Hz, 16-bit Mono PCM, via in-memory streaming or file-based WAV output.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)


class AudioMetadata(BaseModel):
    """Metadata of an audio file."""
    format_name: str
    duration: float
    channels: int
    sample_rate: int
    bit_rate: int | None = None
    size_bytes: int


class AudioProcessorError(Exception):
    """Raised when audio conversion or probing fails."""
    pass


DEFAULT_ENHANCEMENT_FILTER: str = "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"


class AudioProcessor:
    """Universal audio converter and validator using FFmpeg."""

    def __init__(
        self,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
        enhancement_filter: str | None = None,
    ):
        self.ffmpeg_bin = ffmpeg_bin or settings.ffmpeg_bin or "ffmpeg"
        self.ffprobe_bin = ffprobe_bin or settings.ffprobe_bin or "ffprobe"
        self.enhancement_filter = (
            enhancement_filter
            or getattr(settings, "audio_enhancement_filter", None)
            or DEFAULT_ENHANCEMENT_FILTER
        )

    def probe_media(self, file_path: Path) -> AudioMetadata:
        """Probe media file properties using ffprobe with ffmpeg fallback."""
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise AudioProcessorError(f"Input file not found: {file_path}")

        try:
            cmd = [
                self.ffprobe_bin,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-select_streams", "a:0",
                str(file_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data: dict[str, Any] = json.loads(result.stdout)
            format_info = data.get("format", {})
            streams = data.get("streams", [])

            if not streams:
                raise AudioProcessorError(f"No audio stream found in media file: {file_path.name}")

            audio_stream = streams[0]
            duration = float(format_info.get("duration") or audio_stream.get("duration") or 0.0)
            channels = int(audio_stream.get("channels", 1))
            sample_rate = int(audio_stream.get("sample_rate", 16000))
            bit_rate = int(format_info.get("bit_rate")) if format_info.get("bit_rate") else None
            size_bytes = int(format_info.get("size", file_path.stat().st_size))

            return AudioMetadata(
                format_name=format_info.get("format_name", "unknown"),
                duration=duration,
                channels=channels,
                sample_rate=sample_rate,
                bit_rate=bit_rate,
                size_bytes=size_bytes,
            )
        except Exception:
            return self._fallback_probe(file_path)

    def _fallback_probe(self, file_path: Path) -> AudioMetadata:
        """Fallback probe using ffmpeg -i when ffprobe output is unavailable."""
        cmd = [self.ffmpeg_bin, "-i", str(file_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        stderr = res.stderr

        duration = 0.0
        channels = 1
        sample_rate = 16000

        for line in stderr.splitlines():
            line = line.strip()
            if "Duration:" in line:
                try:
                    dur_str = line.split("Duration:")[1].split(",")[0].strip()
                    parts = dur_str.split(":")
                    duration = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                except Exception:
                    pass
            if "Audio:" in line:
                if "Hz" in line:
                    try:
                        for token in line.split(","):
                            if "Hz" in token:
                                sample_rate = int(token.replace("Hz", "").strip())
                            if "stereo" in token:
                                channels = 2
                            elif "mono" in token:
                                channels = 1
                    except Exception:
                        pass

        return AudioMetadata(
            format_name=file_path.suffix.lstrip(".").lower() or "audio",
            duration=duration,
            channels=channels,
            sample_rate=sample_rate,
            size_bytes=file_path.stat().st_size,
        )

    def convert_to_pcm_array(
        self,
        input_path: Path,
        enhance_audio: bool = False,
    ) -> tuple[Any, AudioMetadata]:
        """Stream decoded audio directly from FFmpeg stdout into a normalized float32 NumPy array.

        Eliminates intermediate disk writes for engines supporting in-memory buffers.
        When enhance_audio=True, applies high-pass rumble removal, adaptive FFT noise reduction,
        and EBU R128 loudness normalization before streaming.
        """
        if not HAS_NUMPY or np is None:
            raise AudioProcessorError("NumPy is not installed. Use file-based convert_to_whisper_wav instead.")

        input_path = Path(input_path).resolve()
        metadata = self.probe_media(input_path)

        def _execute_decode(with_filter: bool) -> bytes:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(input_path),
                "-vn",
            ]
            if with_filter and self.enhancement_filter:
                cmd.extend(["-af", self.enhancement_filter])
            cmd.extend([
                "-ar", "16000",
                "-ac", "1",
                "-f", "s16le",
                "-",
            ])
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw, err = proc.communicate()
            if proc.returncode != 0:
                raise AudioProcessorError(f"FFmpeg streaming decode failed: {err.decode(errors='ignore')}")
            return raw

        try:
            raw_pcm = _execute_decode(with_filter=enhance_audio)
        except AudioProcessorError as e:
            if enhance_audio:
                logger.warning(
                    "FFmpeg audio enhancement failed (%s); falling back to standard decode without filter.",
                    e,
                )
                raw_pcm = _execute_decode(with_filter=False)
            else:
                raise

        if not raw_pcm:
            raise AudioProcessorError("Decoded audio stream is empty.")

        # Convert signed 16-bit PCM buffer to normalized float32 [-1.0, 1.0]
        audio_array = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_array, metadata

    def convert_to_whisper_wav(
        self,
        input_path: Path,
        output_path: Path | None = None,
        overwrite: bool = True,
        enhance_audio: bool = False,
    ) -> Path:
        """Convert input media to standard 16kHz mono 16-bit PCM WAV on disk.

        When enhance_audio=True, applies high-pass rumble removal, adaptive FFT noise reduction,
        and EBU R128 loudness normalization.
        """
        input_path = Path(input_path).resolve()
        if not input_path.is_file():
            raise AudioProcessorError(f"Input media file does not exist: {input_path}")

        if output_path is None:
            suffix = "_enhanced.wav" if enhance_audio else "_whisper.wav"
            output_path = input_path.parent / f"{input_path.stem}{suffix}"
        else:
            output_path = Path(output_path).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            return output_path

        def _execute_convert(with_filter: bool):
            cmd = [
                self.ffmpeg_bin,
                "-y" if overwrite else "-n",
                "-i", str(input_path),
                "-vn",
            ]
            if with_filter and self.enhancement_filter:
                cmd.extend(["-af", self.enhancement_filter])
            cmd.extend([
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(output_path),
            ])
            subprocess.run(cmd, capture_output=True, text=True, check=True)

        try:
            _execute_convert(with_filter=enhance_audio)
        except subprocess.CalledProcessError as e:
            if enhance_audio:
                logger.warning(
                    "FFmpeg audio enhancement filter failed (%s); retrying without filter.",
                    e.stderr or str(e),
                )
                try:
                    _execute_convert(with_filter=False)
                except subprocess.CalledProcessError as fallback_err:
                    error_details = fallback_err.stderr or fallback_err.stdout or str(fallback_err)
                    raise AudioProcessorError(f"FFmpeg conversion failed: {error_details}") from fallback_err
            else:
                error_details = e.stderr or e.stdout or str(e)
                raise AudioProcessorError(f"FFmpeg conversion failed: {error_details}") from e

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise AudioProcessorError(f"Converted audio file is empty or missing: {output_path}")

        return output_path
