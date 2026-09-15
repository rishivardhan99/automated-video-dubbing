from pathlib import Path
import subprocess
import wave

from src.utils.logger import get_logger
from src.utils.paths import AUDIO_DIR, OUTPUT_DIR

logger = get_logger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


def extract_audio(video_path: Path) -> Path:
    if not video_path.exists():
        raise AudioExtractionError(
            f"Video file does not exist: {video_path}"
        )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_DIR / f"{video_path.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]

    logger.info("Extracting audio...")
    logger.info("Input: %s", video_path)
    logger.info("Output: %s", audio_path)

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except FileNotFoundError as exc:
        raise AudioExtractionError(
            "FFmpeg was not found. Make sure FFmpeg is installed "
            "and available on PATH."
        ) from exc

    if result.returncode != 0:
        raise AudioExtractionError(
            "FFmpeg failed to extract audio.\n"
            f"{result.stderr}"
        )

    if not audio_path.exists():
        raise AudioExtractionError(
            "FFmpeg completed successfully, but the audio file "
            "was not created."
        )

    logger.info("Audio extraction complete.")

    return audio_path

def get_audio_duration(audio_path: Path) -> float:
    """Return the duration of a WAV file in seconds."""
    if not audio_path.exists():
        return 0.0
    try:
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except wave.Error as exc:
        logger.warning("Could not read duration from %s: %s", audio_path, exc)
        return 0.0

def extract_benchmark_audio(audio_path: Path, max_duration_s: int = 60) -> Path:
    """
    Extract the first N seconds of an audio file for benchmarking.
    Returns the path to the temporary benchmark WAV file.
    """
    if not audio_path.exists():
        raise AudioExtractionError(f"Audio file does not exist: {audio_path}")
        
    benchmark_path = AUDIO_DIR / f"{audio_path.stem}_benchmark.wav"
    
    command = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-t", str(max_duration_s),
        "-c", "copy",
        str(benchmark_path)
    ]
    
    logger.info("Extracting %ds benchmark clip...", max_duration_s)
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioExtractionError("FFmpeg was not found on PATH.") from exc
        
    if result.returncode != 0:
        raise AudioExtractionError(f"Failed to extract benchmark audio:\n{result.stderr}")
        
    if not benchmark_path.exists():
        raise AudioExtractionError("Benchmark audio was not created.")
        
    return benchmark_path

def mux_audio_video(video_path: Path, audio_path: Path) -> Path:
    """Mux a video file and an audio file together."""
    if not video_path.exists():
        raise AudioExtractionError(f"Video file does not exist: {video_path}")
    if not audio_path.exists():
        raise AudioExtractionError(f"Audio file does not exist: {audio_path}")
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{video_path.stem}_dubbed.mp4"
    
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0", # Use video stream from first input
        "-map", "1:a:0", # Use audio stream from second input
        str(output_path)
    ]
    
    logger.info("Muxing final video...")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioExtractionError("FFmpeg was not found on PATH.") from exc
        
    if result.returncode != 0:
        raise AudioExtractionError(f"Failed to mux video:\n{result.stderr}")
        
    if not output_path.exists():
        raise AudioExtractionError("Muxed video was not created.")
        
    return output_path