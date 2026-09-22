import difflib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.utils.logger import get_logger
from src.utils.paths import AUDIO_DIR, STT_CHUNKS_DIR, TRANSCRIPTS_DIR

load_dotenv()

logger = get_logger(__name__)

# Groq's audio file upload limit is 25 MB.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Default chunk duration in seconds for splitting large files.
DEFAULT_CHUNK_DURATION_S = 60  # 1 minute
CHUNK_OVERLAP_S = 2.0  # 2 seconds overlap


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

        try:
            self.max_concurrency = int(os.getenv("STT_MAX_CONCURRENCY", "2"))
        except ValueError:
            self.max_concurrency = 2

        self.client = Groq(api_key=api_key)

    def transcribe(self, audio_path: Path) -> Path:
        """
        Transcribe a full audio file and save the
        normalized transcript JSON.
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

        start_time = time.time()

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

        processing_time = time.time() - start_time
        duration = transcript.get("duration", 0)
        
        # Add metadata for auditable benchmarks
        transcript["processing_time_s"] = round(processing_time, 2)
        transcript["real_time_factor"] = round(processing_time / duration, 3) if duration else 0

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

        max_retries = 5
        for attempt in range(1, max_retries + 1):
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
                    return self._normalize_response(response)

            except Exception as exc:
                message = str(exc).lower()
                
                # Check if it's a standard missing file or something unrecoverable
                if "no such file" in message:
                    raise TranscriptionError(f"File not found: {audio_path}") from exc

                logger.warning(
                    "STT attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                
                if attempt >= max_retries:
                    raise TranscriptionError(
                        f"Groq STT API call failed after {max_retries} attempts: {exc}"
                    ) from exc
                    
                # Backoff logic
                base_delay = min(2 ** (attempt - 1), 30)
                wait_seconds = float(base_delay)
                
                # Try to parse retry-after header if exposed in the exception string
                for token in ["retry-after:", "retry after", "x-ratelimit-reset"]:
                    if token in message:
                        try:
                            after = message.split(token, 1)[1].split()[0]
                            # Remove non-numeric chars
                            after = ''.join(c for c in after if c.isdigit() or c == '.')
                            wait_seconds = max(float(after), base_delay)
                        except (ValueError, IndexError):
                            pass
                            
                # Add small jitter to avoid thundering herd
                import random
                wait_seconds += random.uniform(0.1, 1.0)
                            
                logger.info("Waiting %.2f seconds before STT retry...", wait_seconds)
                time.sleep(wait_seconds)

    def _normalize_response(
        self,
        response,
        time_offset: float = 0.0,
        start_id: int = 0,
    ) -> dict:
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
                start = float(getattr(raw, "start", 0.0) or 0.0)
                end = float(getattr(raw, "end", 0.0) or 0.0)

            if not text:
                continue

            segments.append(
                {
                    "id": start_id + index,
                    "start": round(start + time_offset, 3),
                    "end": round(end + time_offset, 3),
                    "text": text,
                }
            )

        if not segments and full_text.strip():
            segments.append(
                {
                    "id": start_id,
                    "start": round(time_offset, 3),
                    "end": round(time_offset + (duration or 0.0), 3),
                    "text": full_text.strip(),
                }
            )

        transcript = {
            "model": self.model_name,
            "provider": "groq",
            "language": language,
            "language_probability": None,
            "duration": duration,
            "text": " ".join(seg["text"] for seg in segments),
            "segments": segments,
        }

        return transcript

    def _process_chunk_job(self, job: dict) -> dict:
        """
        Worker function to process a single chunk, with caching.
        """
        chunk_path = Path(job["path"])
        cache_path = Path(job["cache_path"])
        chunk_index = job["chunk_index"]
        
        # Check cache
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if "segments" in cached_data:
                    logger.info("Loaded chunk %d from cache", chunk_index)
                    # We still need to return the job context along with result
                    return {"job": job, "result": cached_data, "cached": True}
            except Exception as e:
                logger.warning("Failed to load chunk cache %s: %s", cache_path, e)
                
        logger.info("Transcribing chunk %d (offset %.1fs)...", chunk_index, job["start_offset"])
        result = self._transcribe_single(chunk_path)
        
        # Save cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save chunk cache %s: %s", cache_path, e)

        return {"job": job, "result": result, "cached": False}

    def _is_duplicate_overlap(self, seg: dict, next_chunk_start: float, next_chunk_segments: list) -> bool:
        """
        Determine if a segment is a duplicate caused by overlap boundary.
        """
        # If segment ends well before the overlap starts, it's not a duplicate.
        if seg["end"] < next_chunk_start - 1.0:
            return False
            
        # We need to find if there is a similar segment in the start of the next chunk.
        # We look at the first few segments of the next chunk.
        for next_seg in next_chunk_segments[:5]:
            # If timestamps are very close (within 1 second)
            time_diff = abs(seg["start"] - next_seg["start"])
            if time_diff > 3.0:
                continue
                
            # Check text similarity
            text1 = seg["text"].lower()
            text2 = next_seg["text"].lower()
            
            # Exact match or very high similarity
            if text1 == text2:
                return True
                
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            if similarity > 0.8:
                return True
                
        return False

    def _transcribe_chunked(
        self,
        audio_path: Path,
    ) -> dict:
        """
        Split audio into chunks, transcribe concurrently, and merge results
        handling overlap deduplication.
        """
        chunks = self._split_audio(audio_path)
        
        logger.info("Split into %d chunks. Starting thread pool with %d workers...", 
                    len(chunks), self.max_concurrency)

        video_id = audio_path.stem
        chunk_cache_dir = STT_CHUNKS_DIR / video_id
        chunk_cache_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        for i, chunk_info in enumerate(chunks):
            chunk_path, offset_s, core_duration = chunk_info
            jobs.append({
                "chunk_index": i,
                "start_offset": offset_s,
                "core_duration": core_duration,
                "path": str(chunk_path),
                "cache_path": str(chunk_cache_dir / f"chunk_{i:03d}.json")
            })

        completed_jobs = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_job = {executor.submit(self._process_chunk_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                try:
                    result_data = future.result()
                    completed_jobs.append(result_data)
                except Exception as exc:
                    job = future_to_job[future]
                    logger.error("Chunk %d generated an exception: %s", job["chunk_index"], exc)
                    raise

        # Clean up temporary chunk audio files
        for job in jobs:
            cp = Path(job["path"])
            if cp.exists():
                cp.unlink()
                
        # Sort results by original chunk_index
        completed_jobs.sort(key=lambda x: x["job"]["chunk_index"])

        all_segments = []
        merged_language = None
        merged_duration = 0.0
        
        # Apply time offsets and merge
        for i, completed in enumerate(completed_jobs):
            job = completed["job"]
            partial = completed["result"]
            offset_s = job["start_offset"]
            core_duration = job["core_duration"]
            
            if merged_language is None and partial.get("language"):
                merged_language = partial.get("language")
                
            merged_duration += partial.get("duration", 0.0) or 0.0

            raw_segments = partial.get("segments", [])
            
            # Look ahead for deduplication
            next_chunk_segments = []
            if i + 1 < len(completed_jobs):
                next_partial = completed_jobs[i + 1]["result"]
                next_offset = completed_jobs[i + 1]["job"]["start_offset"]
                for ns in next_partial.get("segments", []):
                    next_chunk_segments.append({
                        "start": ns["start"] + next_offset,
                        "end": ns["end"] + next_offset,
                        "text": ns["text"]
                    })
                    
            next_chunk_start = offset_s + core_duration
            
            for seg in raw_segments:
                seg_start = round(seg["start"] + offset_s, 3)
                seg_end = round(seg["end"] + offset_s, 3)
                seg_text = seg["text"]
                
                adjusted_seg = {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text
                }
                
                # Check if this segment is just overlap duplicate
                if self._is_duplicate_overlap(adjusted_seg, next_chunk_start, next_chunk_segments):
                    logger.debug("Dropped duplicate overlap segment: '%s'", seg_text)
                    continue
                    
                all_segments.append(adjusted_seg)

        # Sort all segments globally by start time to fix timestamp jitter
        all_segments.sort(key=lambda x: x["start"])

        # Renumber all segment IDs sequentially
        for idx, seg in enumerate(all_segments):
            seg["id"] = idx

        transcript = {
            "model": self.model_name,
            "provider": "groq",
            "language": merged_language,
            "chunk_count": len(jobs),
            "completed_chunks": len(completed_jobs),
            "duration": round(merged_duration, 2),
            "text": " ".join(seg["text"] for seg in all_segments),
            "segments": all_segments,
        }

        return transcript

    def _split_audio(
        self,
        audio_path: Path,
    ) -> list[tuple[Path, float, float]]:
        """
        Split an audio file into sequential chunks using FFmpeg.
        Returns a list of (chunk_path, offset_seconds, core_duration) tuples.
        """
        from src.audio import get_audio_duration

        total_duration = get_audio_duration(audio_path)

        if total_duration <= 0:
            raise TranscriptionError("Could not determine audio duration for chunking.")
            
        file_size = audio_path.stat().st_size
        
        # Target ~20MB per chunk for safety
        safe_chunk_size = 20 * 1024 * 1024
        if file_size > safe_chunk_size:
            ratio = safe_chunk_size / file_size
            dynamic_chunk_duration = int(total_duration * ratio)
            actual_chunk_duration = min(self.chunk_duration_s, dynamic_chunk_duration)
        else:
            actual_chunk_duration = self.chunk_duration_s

        chunk_count = math.ceil(total_duration / actual_chunk_duration)
        chunks = []

        for i in range(chunk_count):
            offset = i * actual_chunk_duration
            
            # The extraction duration includes the overlap, but we don't extract past EOF
            extract_duration = actual_chunk_duration + CHUNK_OVERLAP_S
            if offset + extract_duration > total_duration:
                extract_duration = total_duration - offset
                
            chunk_path = AUDIO_DIR / f"{audio_path.stem}_chunk{i:03d}.wav"

            command = [
                "ffmpeg",
                "-y",
                "-i", str(audio_path),
                "-ss", str(offset),
                "-t", str(extract_duration),
                "-c", "copy",
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
                raise TranscriptionError("FFmpeg was not found on PATH.") from exc

            if result.returncode != 0:
                raise TranscriptionError(f"FFmpeg failed to split audio chunk {i}: {result.stderr}")

            if not chunk_path.exists():
                raise TranscriptionError(f"Chunk file not created: {chunk_path}")

            chunks.append((chunk_path, float(offset), float(actual_chunk_duration)))

        return chunks
