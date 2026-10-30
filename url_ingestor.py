"""Direct Media URL & Stream Ingestion Module.

Provides secure HTTP/HTTPS media downloading with SSRF protection, redirect validation,
payload size limits, and optional yt-dlp integration for video/podcast platforms.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import logging
import re
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from config import find_binary, settings

logger = logging.getLogger(__name__)


class URLIngestionError(Exception):
    """Raised when URL validation, SSRF checks, or media downloading fails."""


@dataclass
class DownloadedMedia:
    """Metadata and local file reference for a downloaded media URL."""

    file_path: Path
    display_name: str
    source_url: str
    size_bytes: int
    extractor_used: str


DIRECT_MEDIA_EXTENSIONS: set[str] = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".flac",
    ".webm",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wma",
    ".mpeg",
    ".mpg",
    ".m4v",
}

CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
}

PLATFORM_DOMAINS: tuple[str, ...] = (
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
    "soundcloud.com",
    "www.soundcloud.com",
    "twitch.tv",
    "www.twitch.tv",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "tiktok.com",
    "www.tiktok.com",
    "dailymotion.com",
    "www.dailymotion.com",
    "bilibili.com",
    "www.bilibili.com",
)


def sanitize_filename(name: str, default: str = "url_media.wav") -> str:
    """Sanitize a filename by stripping directory components and unsafe characters."""
    if not name or not name.strip():
        return default
    base = Path(name.strip()).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return cleaned[:180] if cleaned else default


class MediaURLIngestor:
    """Validates and ingests remote audio/video URLs into local workspace files."""

    def __init__(
        self,
        max_download_size_mb: int | None = None,
        timeout_seconds: int | None = None,
        allow_local_network: bool | None = None,
        ytdlp_bin: str | None = None,
    ):
        self.max_download_size_mb = (
            max_download_size_mb
            if max_download_size_mb is not None
            else settings.url_max_download_size_mb
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.url_download_timeout_seconds
        )
        self.allow_local_network = (
            allow_local_network
            if allow_local_network is not None
            else settings.url_allow_local_network
        )
        self.ytdlp_bin = ytdlp_bin or settings.ytdlp_bin or find_binary("yt-dlp")

    @property
    def max_bytes(self) -> int:
        return max(1, self.max_download_size_mb) * 1024 * 1024

    def is_ytdlp_available(self) -> bool:
        """Check whether yt-dlp binary or Python package is available."""
        if self.ytdlp_bin and Path(self.ytdlp_bin).is_file():
            return True
        if find_binary("yt-dlp"):
            return True
        return importlib.util.find_spec("yt_dlp") is not None

    def validate_media_url(self, url: str) -> str:
        """Validate URL syntax, scheme, and SSRF IP address safety."""
        if not url or not isinstance(url, str) or not url.strip():
            raise URLIngestionError("Media URL cannot be empty.")

        clean_url = url.strip()
        try:
            parsed = urlsplit(clean_url)
        except Exception as exc:
            raise URLIngestionError(f"Invalid URL format: {exc}") from exc

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise URLIngestionError(
                f"Unsupported URL scheme '{scheme or 'none'}'. Only http:// and https:// URLs are allowed."
            )

        hostname = parsed.hostname
        if not hostname:
            raise URLIngestionError("URL must include a valid hostname.")

        if not self.allow_local_network:
            self._enforce_public_host(hostname)

        return clean_url

    def _enforce_public_host(self, hostname: str) -> None:
        """Reject localhost, private, loopback, link-local, or reserved IP addresses."""
        lower_host = hostname.lower().rstrip(".")
        if lower_host in ("localhost", "localhost.localdomain") or lower_host.endswith(
            (".local", ".internal", ".localhost")
        ):
            raise URLIngestionError(
                f"Access to local or internal host '{hostname}' is blocked by security policy."
            )

        # Check if hostname is a literal IP address first
        try:
            ip_obj = ipaddress.ip_address(lower_host)
            self._check_ip_allowed(ip_obj, hostname)
            return
        except ValueError:
            pass

        # Resolve DNS and inspect all returned records
        try:
            addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise URLIngestionError(f"Could not resolve hostname '{hostname}': {exc}") from exc

        if not addr_info:
            raise URLIngestionError(f"No IP address found for hostname '{hostname}'.")

        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError as exc:
                raise URLIngestionError(f"Invalid resolved IP '{ip_str}' for '{hostname}'.") from exc
            self._check_ip_allowed(ip_obj, hostname)

    @staticmethod
    def _check_ip_allowed(
        ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str
    ) -> None:
        # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped

        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            raise URLIngestionError(
                f"Access to private or restricted network address ({ip_obj}) for host '{hostname}' is blocked."
            )

    @staticmethod
    def derive_initial_display_name(url: str) -> str:
        """Derive a human-readable placeholder filename from a URL before downloading."""
        try:
            parsed = urlsplit(url.strip())
            path_name = Path(unquote(parsed.path)).name
            if path_name and path_name not in ("/", "."):
                return sanitize_filename(path_name, default="remote_media")
            if parsed.hostname:
                return sanitize_filename(f"{parsed.hostname}_media", default="remote_media")
        except Exception:
            pass
        return "remote_media"

    @staticmethod
    def _extract_content_disposition_filename(header_val: str | None) -> str | None:
        if not header_val:
            return None
        # RFC 5987 filename*=UTF-8''...
        match_utf8 = re.search(r"filename\*\s*=\s*UTF-8''([^;\r\n]+)", header_val, re.IGNORECASE)
        if match_utf8:
            return sanitize_filename(unquote(match_utf8.group(1).strip("\"'")))
        match_std = re.search(r'filename\s*=\s*"([^"]+)"', header_val, re.IGNORECASE)
        if match_std:
            return sanitize_filename(match_std.group(1))
        match_unquoted = re.search(r"filename\s*=\s*([^;\s]+)", header_val, re.IGNORECASE)
        if match_unquoted:
            return sanitize_filename(match_unquoted.group(1).strip("\"'"))
        return None

    def _is_platform_url(self, url: str) -> bool:
        try:
            host = (urlsplit(url).hostname or "").lower()
            return any(host == d or host.endswith("." + d) for d in PLATFORM_DOMAINS)
        except Exception:
            return False

    def download_media(
        self,
        url: str,
        dest_dir: Path,
        job_id: str | None = None,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
    ) -> DownloadedMedia:
        """Validate URL and download audio/video stream into dest_dir."""
        clean_url = self.validate_media_url(url)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{job_id}_" if job_id else ""

        # If known video/social platform URL, use yt-dlp directly when available
        if self._is_platform_url(clean_url):
            if not self.is_ytdlp_available():
                raise URLIngestionError(
                    "This URL requires 'yt-dlp' for platform media extraction, but 'yt-dlp' is not installed."
                )
            return self._download_via_ytdlp(clean_url, dest_dir, prefix, progress_callback)

        return self._download_via_httpx(clean_url, dest_dir, prefix, progress_callback)

    def _download_via_httpx(
        self,
        url: str,
        dest_dir: Path,
        prefix: str,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
    ) -> DownloadedMedia:
        """Stream direct media over HTTP/HTTPS with redirect SSRF checks and size limits."""
        timeout = httpx.Timeout(float(self.timeout_seconds), connect=15.0)
        headers = {
            "User-Agent": "AudioTranscriber/1.1"
        }
        current_url = url
        max_redirects = 5

        with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
            for _ in range(max_redirects + 1):
                with client.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise URLIngestionError(
                                f"Remote server returned redirect (HTTP {response.status_code}) without a Location header."
                            )
                        next_url = urljoin(current_url, location)
                        current_url = self.validate_media_url(next_url)
                        continue

                    if response.status_code != 200:
                        raise URLIngestionError(
                            f"Failed to download media from URL (HTTP {response.status_code})."
                        )

                    content_type = (
                        response.headers.get("content-type", "").split(";")[0].strip().lower()
                    )

                    # If remote server returned an HTML webpage, delegate to yt-dlp if available
                    if content_type in ("text/html", "application/xhtml+xml"):
                        if self.is_ytdlp_available():
                            logger.info(
                                "URL '%s' returned %s; delegating extraction to yt-dlp.",
                                current_url,
                                content_type,
                            )
                            return self._download_via_ytdlp(
                                current_url, dest_dir, prefix, progress_callback
                            )
                        raise URLIngestionError(
                            "The URL returned an HTML webpage instead of an audio/video stream. "
                            "Provide a direct media file link (.mp3, .wav, .mp4, etc.) or install 'yt-dlp' for webpage extraction."
                        )

                    # Check Content-Length header before downloading body
                    content_length_raw = response.headers.get("content-length")
                    total_bytes: int | None = None
                    if content_length_raw and content_length_raw.isdigit():
                        total_bytes = int(content_length_raw)
                        if total_bytes > self.max_bytes:
                            size_mb = total_bytes / (1024 * 1024)
                            raise URLIngestionError(
                                f"Remote file size ({size_mb:.1f} MB) exceeds maximum allowed limit ({self.max_download_size_mb} MB)."
                            )

                    # Determine filename & extension
                    cd_name = self._extract_content_disposition_filename(
                        response.headers.get("content-disposition")
                    )
                    url_path_name = Path(unquote(urlsplit(current_url).path)).name
                    raw_display = cd_name or url_path_name or "remote_audio"

                    ext = Path(raw_display).suffix.lower()
                    if not ext or ext not in DIRECT_MEDIA_EXTENSIONS:
                        ext = CONTENT_TYPE_TO_EXT.get(content_type, ext or ".wav")
                        if not Path(raw_display).suffix:
                            raw_display = f"{raw_display}{ext}"

                    display_name = sanitize_filename(raw_display, default=f"remote_media{ext}")
                    safe_ext = re.sub(r"[^a-zA-Z0-9.]", "", ext)[:10] or ".wav"
                    target_path = dest_dir / f"{prefix}url_download{safe_ext}"

                    downloaded = 0
                    try:
                        with open(target_path, "wb") as f_out:
                            for chunk in response.iter_bytes(chunk_size=65536):
                                if not chunk:
                                    continue
                                downloaded += len(chunk)
                                if downloaded > self.max_bytes:
                                    raise URLIngestionError(
                                        f"Media stream exceeded maximum allowed size limit ({self.max_download_size_mb} MB)."
                                    )
                                f_out.write(chunk)
                                if progress_callback:
                                    progress_callback(downloaded, total_bytes, display_name)
                    except Exception:
                        if target_path.exists():
                            try:
                                target_path.unlink()
                            except Exception:
                                pass
                        raise

                    if downloaded == 0:
                        if target_path.exists():
                            try:
                                target_path.unlink()
                            except Exception:
                                pass
                        raise URLIngestionError("Downloaded media file is empty (0 bytes).")

                    return DownloadedMedia(
                        file_path=target_path,
                        display_name=display_name,
                        source_url=url,
                        size_bytes=downloaded,
                        extractor_used="httpx",
                    )

            raise URLIngestionError(f"Too many redirects (exceeded {max_redirects}) while fetching URL.")

    def _download_via_ytdlp(
        self,
        url: str,
        dest_dir: Path,
        prefix: str,
        progress_callback: Callable[[int, int | None, str], None] | None = None,
    ) -> DownloadedMedia:
        """Extract audio stream using yt-dlp binary or Python module."""
        if progress_callback:
            progress_callback(0, None, self.derive_initial_display_name(url))

        out_template = str(dest_dir / f"{prefix}ytdlp_%(id)s.%(ext)s")
        bin_path = self.ytdlp_bin or find_binary("yt-dlp")

        if bin_path:
            cmd = [
                bin_path,
                "--no-playlist",
                "--no-warnings",
                "--no-progress",
                "--max-filesize",
                f"{self.max_download_size_mb}M",
                "-f",
                "bestaudio/best",
                "-o",
                out_template,
                "--print",
                "after_move:filepath",
                "--print",
                "%(title)s.%(ext)s",
                url,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                self._cleanup_ytdlp_partials(dest_dir, prefix)
                raise URLIngestionError(
                    f"Media extraction timed out after {self.timeout_seconds}s."
                ) from exc

            if proc.returncode != 0:
                self._cleanup_ytdlp_partials(dest_dir, prefix)
                err_msg = (proc.stderr or proc.stdout or "Unknown yt-dlp error").strip()
                raise URLIngestionError(f"yt-dlp extraction failed: {err_msg[:300]}")

            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            file_path: Path | None = None
            display_title = self.derive_initial_display_name(url)

            if len(lines) >= 2:
                display_title = sanitize_filename(lines[-2], default=display_title)
                candidate = Path(lines[-1])
                if candidate.is_file():
                    file_path = candidate
            elif len(lines) == 1:
                candidate = Path(lines[0])
                if candidate.is_file():
                    file_path = candidate

            if file_path is None:
                matches = sorted(dest_dir.glob(f"{prefix}ytdlp_*"))
                if matches:
                    file_path = matches[0]

            if file_path is None or not file_path.is_file():
                raise URLIngestionError("yt-dlp finished but no media file was produced.")

            size_bytes = file_path.stat().st_size
            if size_bytes == 0:
                file_path.unlink(missing_ok=True)
                raise URLIngestionError("Extracted media file is empty.")
            if size_bytes > self.max_bytes:
                file_path.unlink(missing_ok=True)
                raise URLIngestionError(
                    f"Extracted media exceeds maximum allowed size ({self.max_download_size_mb} MB)."
                )

            return DownloadedMedia(
                file_path=file_path,
                display_name=display_title,
                source_url=url,
                size_bytes=size_bytes,
                extractor_used="yt-dlp",
            )

        # Fallback to Python yt_dlp library if installed
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise URLIngestionError("yt-dlp is not installed on this system.") from exc

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": self.max_bytes,
            "socket_timeout": min(30, self.timeout_seconds),
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_str = ydl.prepare_filename(info)
                file_path = Path(downloaded_str)
                title = info.get("title") or "remote_media"
                ext = info.get("ext") or file_path.suffix.lstrip(".") or "mp3"
                display_name = sanitize_filename(f"{title}.{ext}")
        except Exception as exc:
            self._cleanup_ytdlp_partials(dest_dir, prefix)
            raise URLIngestionError(f"yt-dlp extraction failed: {exc}") from exc

        if not file_path.is_file() or file_path.stat().st_size == 0:
            self._cleanup_ytdlp_partials(dest_dir, prefix)
            raise URLIngestionError("yt-dlp failed to produce a valid media file.")

        return DownloadedMedia(
            file_path=file_path,
            display_name=display_name,
            source_url=url,
            size_bytes=file_path.stat().st_size,
            extractor_used="yt-dlp",
        )

    @staticmethod
    def _cleanup_ytdlp_partials(dest_dir: Path, prefix: str) -> None:
        if not prefix:
            return
        for p in dest_dir.glob(f"{prefix}ytdlp_*"):
            try:
                p.unlink()
            except Exception:
                pass
