"""Media acquisition and normalization; no model inference."""

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import soundfile as sf
import yt_dlp

SR = 16000
MAX_DURATION = int(os.getenv("MAX_DURATION_SECONDS", "3600"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def timestamp(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def public_url(url):
    """Reject local/private network destinations and non-web schemes."""
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Use a public HTTP or HTTPS URL without embedded credentials.")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only standard HTTP/HTTPS ports are supported.")
    addresses = socket.getaddrinfo(
        parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    )
    if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
        raise ValueError("Local and private network URLs are not supported.")
    return parsed


def fetch_direct(url, directory):
    # Validate every redirect, impose timeouts, stream to disk, cap download size.
    for _ in range(6):
        public_url(url)
        with requests.get(url, stream=True, allow_redirects=False, timeout=(15, 60)) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The media URL returned an invalid redirect.")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError(
                    "This is a webpage. Use a direct media file URL or upload the file."
                )
            if int(response.headers.get("Content-Length") or 0) > MAX_BYTES:
                raise ValueError("The media file exceeds 500 MB.")
            path = str(Path(directory) / "download.media")
            total = 0
            with open(path, "wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError("The download exceeds 500 MB.")
                    output.write(chunk)
            return path, "Linked recording", None
    raise ValueError("Too many redirects. Upload the file instead.")


def fetch_youtube(url, directory):
    skip_reason = []

    def reject(info, *, incomplete=False):
        if info.get("is_live") or info.get("live_status") == "is_live":
            skip_reason[:] = ["Live streams are not supported."]
            return "Live streams are not supported."
        if (info.get("duration") or 0) > MAX_DURATION:
            reason = f"This video exceeds the {MAX_DURATION // 60}-minute limit. Increase MAX_DURATION_SECONDS or use a shorter recording."
            skip_reason[:] = [reason]
            return reason
        return None

    def limit_download(event):
        if event.get("downloaded_bytes", 0) > MAX_BYTES:
            raise ValueError("The download exceeds 500 MB.")

    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(Path(directory) / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        "max_filesize": MAX_BYTES,
        "match_filter": reject,
        "progress_hooks": [limit_download],
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        try:
            info = downloader.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            message = str(exc).lower()
            if "not a bot" in message or "sign in" in message:
                raise ValueError(
                    "YouTube blocked this server download. Download on your own computer and upload the audio/video; changing the search model will not fix this."
                ) from exc
            raise ValueError(
                "YouTube download failed. Try an uploaded file. Downloader details: "
                + str(exc)[:250]
            ) from exc
        if skip_reason:
            raise ValueError(skip_reason[0])
        if not info or info.get("_type") in {"playlist", "multi_video"}:
            raise ValueError("Paste a link to one video, not a channel or playlist.")
        downloads = info.get("requested_downloads") or []
        path = downloads[0].get("filepath") if downloads else downloader.prepare_filename(info)
        if not path or not Path(path).is_file():
            raise ValueError(
                "YouTube download was blocked or skipped. Upload the recording instead."
            )
        if Path(path).stat().st_size > MAX_BYTES:
            raise ValueError("The download exceeds 500 MB.")
        video_id = info.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Could not identify a YouTube video.")
        return path, info.get("title", "YouTube recording"), video_id


def source_key(upload, url):
    if bool(upload) == bool(url.strip()):
        raise ValueError("Choose one source: upload a file OR enter a URL.")
    if upload:
        stat = os.stat(upload)
        if stat.st_size > MAX_BYTES:
            raise ValueError("Upload a file smaller than 500 MB.")
        return ("file", upload, stat.st_size, stat.st_mtime_ns)
    return ("url", url.strip())


def prepare_source(upload, url, progress):
    if shutil.which("ffmpeg") is None:
        raise ValueError("FFmpeg is missing. Run the setup cell or add ffmpeg to packages.txt.")
    directory = tempfile.mkdtemp(prefix="media_search_")
    try:
        progress(0.02, desc="Reading or downloading media…")
        if upload:
            source, title, video_id = upload, Path(upload).name, None
        else:
            parsed = public_url(url.strip())
            if parsed.hostname.lower() in YOUTUBE_HOSTS:
                source, title, video_id = fetch_youtube(url.strip(), directory)
            else:
                source, title, video_id = fetch_direct(url.strip(), directory)
        progress(0.08, desc="Extracting audio…")
        wav = str(Path(directory) / "recording.wav")
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                source,
                "-vn",
                "-t",
                str(MAX_DURATION + 1),
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-c:a",
                "pcm_s16le",
                wav,
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode:
            raise ValueError("Cannot decode the media. Try uploading an MP3, WAV, M4A, or MP4.")
        info = sf.info(wav)
        if info.frames == 0:
            raise ValueError("The media contains no audio.")
        if info.duration > MAX_DURATION:
            raise ValueError(
                f"The recording exceeds the {MAX_DURATION // 60}-minute limit. Increase MAX_DURATION_SECONDS or use a shorter recording."
            )
        # Only the decoded WAV is needed after indexing.
        if not upload:
            Path(source).unlink(missing_ok=True)
        return {
            "directory": directory,
            "wav": wav,
            "duration": info.duration,
            "frames": info.frames,
            "title": title,
            "video_id": video_id,
        }
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
