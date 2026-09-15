import json
import math
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.utils.logger import get_logger
from src.utils.paths import TRANSCRIPTS_DIR, AUDIO_DIR

load_dotenv()

logger = get_logger(__name__)

# Groq's audio file upload limit is 25 MB.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Default chunk duration in seconds for splitting large files.
DEFAULT_CHUNK_DURATION_S = 600  # 10 minutes


class TranscriptionError(Exception):
    """Raised when transcription fails."""


class GroqTranscriber:
    """Transcribe audio using the Groq Speech-to-Text API."""

    def __init__(
        self,
        model_name: str | None = None,
        language: str | None = None,
        chunk_duration_s: int | None = None,
    ) -> None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise TranscriptionError(
                "GROQ_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.model_name = (
            model_name
            or os.getenv(
                "GROQ_STT_MODEL",
                "whisper-large-v3-turbo",
            )
        )

        self.language = (
            language
            or os.getenv("GROQ_STT_LANGUAGE")
            or None
        )

        self.chunk_duration_s = (
            chunk_duration_s
            or DEFAULT_CHUNK_DURATION_S
        )

        self.client = Groq(api_key=api_key)

    def transcribe(self, audio_path: Path) -> Path:
        """
        Transcribe a full audio file and save the
        normalized transcript JSON.

        If the file exceeds Groq's upload limit, it is
        automatically split into sequential chunks,
        transcribed independently, and merged.
        """

        if not audio_path.exists():
            raise TranscriptionError(
                f"Audio file does not exist: {audio_path}"
            )

        TRANSCRIPTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        transcript_path = (
            TRANSCRIPTS_DIR
            / f"{audio_path.stem}.json"
        )

        file_size = audio_path.stat().st_size

        logger.info(
            "Audio file size: %.1f MB",
            file_size / (1024 * 1024),
        )

        if file_size <= MAX_UPLOAD_BYTES:
            transcript = self._transcribe_single(
                audio_path,
            )
        else:
            logger.info(
                "File exceeds %.0f MB limit. "
                "Splitting into chunks...",
                MAX_UPLOAD_BYTES / (1024 * 1024),
            )
            transcript = self._transcribe_chunked(
                audio_path,
            )

        with transcript_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                transcript,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Transcription complete.")
        logger.info(
            "Detected language: %s",
            transcript.get("language"),
        )
        logger.info(
            "Segments: %d",
            len(transcript.get("segments", [])),
        )
        logger.info(
            "Saved transcript: %s",
            transcript_path,
        )

        return transcript_path

    def _transcribe_single(
        self,
        audio_path: Path,
    ) -> dict:
        """
        Transcribe a single audio file that fits within
        the Groq upload limit.
        """

        logger.info(
            "Transcribing with Groq STT: %s",
            audio_path.name,
        )
        logger.info(
            "Model: %s | Language: %s",
            self.model_name,
            self.language or "auto-detect",
        )

        try:
            with audio_path.open("rb") as audio_file:

                kwargs = {
                    "model": self.model_name,
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                }

                if self.language:
                    kwargs["language"] = self.language

                response = (
                    self.client.audio
                    .transcriptions
                    .create(**kwargs)
                )

        except Exception as exc:
            raise TranscriptionError(
                f"Groq STT API call failed: {exc}"
            ) from exc

        return self._normalize_response(response)

    def _normalize_response(
        self,
        response,
        time_offset: float = 0.0,
        start_id: int = 0,
    ) -> dict:
        """
        Normalize a Groq verbose_json response into our
        internal transcript structure.
        """

        # The verbose_json response may be a Pydantic
        # model or a dict-like object.
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "__dict__"):
            data = vars(response)
        else:
            data = dict(response)

        raw_segments = data.get("segments", [])
        full_text = data.get("text", "")
        language = data.get("language")
        duration = data.get("duration")

        segments = []

        for index, raw in enumerate(
            raw_segments if raw_segments else [],
        ):
            text = ""
            start = 0.0
            end = 0.0

            if isinstance(raw, dict):
                text = raw.get("text", "").strip()
                start = float(raw.get("start", 0.0))
                end = float(raw.get("end", 0.0))
            elif hasattr(raw, "text"):
                text = (raw.text or "").strip()
                start = float(
                    getattr(raw, "start", 0.0) or 0.0
                )
                end = float(
                    getattr(raw, "end", 0.0) or 0.0
                )

            if not text:
                continue

            segments.append(
                {
                    "id": start_id + index,
                    "start": round(
                        start + time_offset, 3
                    ),
                    "end": round(
                        end + time_offset, 3
                    ),
                    "text": text,
                }
            )

        # If no segments were parsed but we have text,
        # create a single segment from the full text.
        if not segments and full_text.strip():
            segments.append(
                {
                    "id": start_id,
                    "start": round(time_offset, 3),
                    "end": round(
                        time_offset
                        + (duration or 0.0),
                        3,
                    ),
                    "text": full_text.strip(),
                }
            )

        transcript = {
            "model": self.model_name,
            "provider": "groq",
            "language": language,
            "language_probability": None,
            "duration": duration,
            "text": " ".join(
                seg["text"] for seg in segments
            ),
            "segments": segments,
        }

        return transcript

    def _transcribe_chunked(
        self,
        audio_path: Path,
    ) -> dict:
        """
        Split audio into chunks, transcribe each via
        Groq, and merge results.
        """

        chunks = self._split_audio(audio_path)

        logger.info(
            "Split into %d chunks.",
            len(chunks),
        )

        all_segments = []
        merged_language = None
        merged_duration = 0.0
        current_id = 0

        try:
            for chunk_index, (
                chunk_path,
                offset_s,
            ) in enumerate(chunks, start=1):

                logger.info(
                    "Transcribing chunk %d/%d "
                    "(offset %.1fs)...",
                    chunk_index,
                    len(chunks),
                    offset_s,
                )

                partial = self._transcribe_single(
                    chunk_path,
                )

                # Apply time offset to segments.
                for seg in partial.get(
                    "segments", []
                ):
                    seg["start"] = round(
                        seg["start"] + offset_s, 3
                    )
                    seg["end"] = round(
                        seg["end"] + offset_s, 3
                    )
                    seg["id"] = current_id
                    current_id += 1

                all_segments.extend(
                    partial.get("segments", [])
                )

                if (
                    merged_language is None
                    and partial.get("language")
                ):
                    merged_language = partial[
                        "language"
                    ]

                merged_duration += partial.get(
                    "duration", 0.0
                ) or 0.0

        finally:
            # Clean up temporary chunk files.
            for chunk_path, _ in chunks:
                if chunk_path.exists():
                    chunk_path.unlink()
                    logger.info(
                        "Cleaned up chunk: %s",
                        chunk_path.name,
                    )

        transcript = {
            "model": self.model_name,
            "provider": "groq",
            "language": merged_language,
            "language_probability": None,
            "duration": merged_duration,
            "text": " ".join(
                seg["text"]
                for seg in all_segments
            ),
            "segments": all_segments,
        }

        return transcript

    def _split_audio(
        self,
        audio_path: Path,
    ) -> list[tuple[Path, float]]:
        """
        Split an audio file into sequential chunks
        using FFmpeg.

        Returns a list of (chunk_path, offset_seconds)
        tuples.
        """

        from src.audio import get_audio_duration

        total_duration = get_audio_duration(audio_path)

        if total_duration <= 0:
            raise TranscriptionError(
                "Could not determine audio duration "
                "for chunking."
            )
            
        file_size = audio_path.stat().st_size
        
        # Target ~20MB per chunk for safety
        safe_chunk_size = 20 * 1024 * 1024
        if file_size > safe_chunk_size:
            ratio = safe_chunk_size / file_size
            dynamic_chunk_duration = int(total_duration * ratio)
            actual_chunk_duration = min(self.chunk_duration_s, dynamic_chunk_duration)
        else:
            actual_chunk_duration = self.chunk_duration_s

        chunk_count = math.ceil(
            total_duration / actual_chunk_duration
        )

        chunks = []

        for i in range(chunk_count):
            offset = i * actual_chunk_duration

            chunk_path = (
                AUDIO_DIR
                / f"{audio_path.stem}_chunk{i:03d}.wav"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-ss",
                str(offset),
                "-t",
                str(actual_chunk_duration),
                "-c",
                "copy",
                str(chunk_path),
            ]

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
                raise TranscriptionError(
                    "FFmpeg was not found on PATH."
                ) from exc

            if result.returncode != 0:
                raise TranscriptionError(
                    "FFmpeg failed to split audio "
                    f"chunk {i}: {result.stderr}"
                )

            if not chunk_path.exists():
                raise TranscriptionError(
                    f"Chunk file not created: "
                    f"{chunk_path}"
                )

            chunks.append((chunk_path, float(offset)))

        return chunks
