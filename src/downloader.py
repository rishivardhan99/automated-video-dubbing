from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from src.utils.logger import get_logger
from src.utils.paths import INPUT_DIR

logger = get_logger(__name__)


class DownloadError(Exception):
    """Raised when a video cannot be downloaded."""


def validate_youtube_url(url: str) -> bool:
    """
    Validate that the supplied URL looks like a supported YouTube URL.
    """

    try:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        hostname = (parsed.hostname or "").lower()

        return hostname in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }

    except ValueError:
        return False


def download_video(url: str) -> Path:
    """
    Download a YouTube video and return the resulting local file path.
    """

    if not validate_youtube_url(url):
        raise DownloadError(
            "Invalid YouTube URL. Please provide a valid YouTube URL."
        )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_template = str(INPUT_DIR / "%(title)s.%(ext)s")

    base_options = {
        # Prefer a reasonable MP4 video + M4A audio combination.
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
            "best[ext=mp4]/"
            "best"
        ),

        "outtmpl": output_template,

        # Merge separate video/audio streams when necessary.
        "merge_output_format": "mp4",

        # Don't overwrite an existing downloaded video.
        "nooverwrites": True,

        # Show yt-dlp's progress information.
        "progress": True,

        # Keep the downloader relatively quiet.
        "quiet": False,

        # Don't download playlists accidentally.
        "noplaylist": True,
    }

    logger.info("Starting video download...")
    logger.info("URL: %s", url)

    strategies = []
    
    if Path("cookies.txt").exists():
        strategies.append(("cookies.txt", {"cookiefile": "cookies.txt"}))
        
    strategies.extend([
        ("Default Client", {}),
        ("Edge Cookies", {"cookiesfrombrowser": ("edge", None, None, None)}),
        ("Chrome Cookies", {"cookiesfrombrowser": ("chrome", None, None, None)}),
        ("Firefox Cookies", {"cookiesfrombrowser": ("firefox", None, None, None)}),
    ])

    last_error = None

    for strategy_name, extra_options in strategies:
        options = base_options.copy()
        options.update(extra_options)

        if extra_options:
            logger.info("Retrying download using strategy: %s...", strategy_name)

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is None:
                    raise DownloadError("yt-dlp returned no video information.")

                downloaded_path = Path(ydl.prepare_filename(info))

                # If separate streams were merged into MP4,
                # yt-dlp may have changed the final extension.
                if not downloaded_path.exists():
                    possible_mp4 = downloaded_path.with_suffix(".mp4")

                    if possible_mp4.exists():
                        downloaded_path = possible_mp4

                if not downloaded_path.exists():
                    raise DownloadError(
                        "Download appeared to succeed, but the output file "
                        "could not be found."
                    )

                logger.info("Download complete.")
                logger.info("Saved to: %s", downloaded_path)

                return downloaded_path

        except Exception as exc:
            logger.warning("Strategy '%s' failed: %s", strategy_name, exc)
            last_error = exc

    raise DownloadError(
        f"All download strategies failed. YouTube bot detection blocked the download. Last error: {last_error}"
    ) from last_error