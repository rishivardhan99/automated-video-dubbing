import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from pydub import AudioSegment
import edge_tts
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.paths import AUDIO_DIR
from src.audio import get_audio_duration

load_dotenv()

logger = get_logger(__name__)


class SynthesisError(Exception):
    """Raised when audio synthesis fails."""


@dataclass
class SegmentMetadata:
    id: int
    start: float
    end: float
    target_duration: float
    generated_duration: float
    speed_factor: float
    applied_speed_factor: float
    timing_violation: bool
    generated_audio_path: str


class EdgeTTSSynthesizer:
    def __init__(
        self,
        voice: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.voice = voice or os.getenv("EDGE_TTS_VOICE", "en-US-GuyNeural")
        self.max_retries = max_retries

    def synthesize(
        self,
        translation_path: Path,
        source_audio_path: Path,
    ) -> tuple[Path, list[SegmentMetadata]]:
        """
        Generate TTS for all translated segments and assemble them
        onto a canvas matching the exact duration of the source audio.
        """
        
        if not translation_path.exists():
            raise SynthesisError(f"Translation JSON not found: {translation_path}")
            
        if not source_audio_path.exists():
            raise SynthesisError(f"Source audio not found: {source_audio_path}")

        # Ensure audio dir exists
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        
        source_duration_s = get_audio_duration(source_audio_path)
        if source_duration_s <= 0:
            raise SynthesisError("Could not determine source audio duration.")

        with translation_path.open("r", encoding="utf-8") as f:
            translation_data = json.load(f)

        segments = translation_data.get("segments", [])
        if not segments:
            raise SynthesisError("Translation contains no segments.")
            
        logger.info("Initializing synthesis canvas of duration %.2fs", source_duration_s)
        
        metadata_report = []
        
        # Pydub works in milliseconds
        canvas_duration_ms = int(source_duration_s * 1000)
        canvas = AudioSegment.silent(duration=canvas_duration_ms)
        
        # Base prefix for this job
        job_prefix = translation_path.stem
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            
            # Run the async generation and assembly
            canvas, metadata_report = asyncio.run(
                self._process_all_segments(
                    segments, 
                    canvas, 
                    job_prefix,
                    temp_path,
                )
            )
        
        dubbed_dir = AUDIO_DIR / "dubbed"
        dubbed_dir.mkdir(parents=True, exist_ok=True)
        output_path = dubbed_dir / f"{job_prefix}_dubbed.wav"
        
        logger.info("Exporting final assembled audio to %s", output_path)
        canvas.export(str(output_path), format="wav")
        
        return output_path, metadata_report

    async def _process_all_segments(
        self,
        segments: list[dict],
        canvas: AudioSegment,
        job_prefix: str,
        output_dir: Path,
    ) -> tuple[AudioSegment, list[SegmentMetadata]]:
        
        metadata_report = []
        
        for segment in segments:
            seg_id = segment["id"]
            start_s = segment["start"]
            end_s = segment["end"]
            text = segment["translated_text"]
            
            target_duration_s = end_s - start_s
            
            logger.info(
                "Synthesizing segment %d (%.2fs - %.2fs, target duration: %.2fs)", 
                seg_id, start_s, end_s, target_duration_s
            )
            
            raw_clip_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_raw.mp3"
            
            # 1. Generate Raw TTS
            if not text.strip():
                logger.info("Segment %d text is empty. Creating silent placeholder.", seg_id)
                audio_segment = AudioSegment.silent(duration=0)
                # Create an empty file just in case it's needed for metadata
                raw_clip_path.touch()
            else:
                await self._generate_tts_with_retry(text, raw_clip_path)
                audio_segment = AudioSegment.from_file(str(raw_clip_path))
            
            # 2. Measure actual duration
            actual_duration_s = len(audio_segment) / 1000.0
            
            # 3. Calculate speed factor and decide action
            # speed_factor = how much we need to speed it up to fit in target_duration
            if target_duration_s <= 0:
                speed_factor = 1.0 # Protect against zero div
            else:
                speed_factor = actual_duration_s / target_duration_s
                
            applied_factor = 1.0
            violation = False
            
            if speed_factor <= 1.05:
                # Leave unchanged
                final_clip_path = raw_clip_path
                applied_factor = 1.0
            elif speed_factor <= 1.25:
                # Apply exact required atempo
                applied_factor = speed_factor
                final_clip_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_stretched.wav"
                self._apply_atempo(raw_clip_path, final_clip_path, applied_factor)
                audio_segment = AudioSegment.from_file(str(final_clip_path))
            else:
                # Cap at 1.25 and flag violation
                applied_factor = 1.25
                violation = True
                final_clip_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_stretched.wav"
                self._apply_atempo(raw_clip_path, final_clip_path, applied_factor)
                audio_segment = AudioSegment.from_file(str(final_clip_path))
                
                logger.warning(
                    "[TIMING VIOLATION] Segment %d generated %.2fs audio for %.2fs target. "
                    "Max 1.25x stretch applied.",
                    seg_id, actual_duration_s, target_duration_s
                )
                
            # 4. Record metadata
            meta = SegmentMetadata(
                id=seg_id,
                start=start_s,
                end=end_s,
                target_duration=target_duration_s,
                generated_duration=actual_duration_s,
                speed_factor=speed_factor,
                applied_speed_factor=applied_factor,
                timing_violation=violation,
                generated_audio_path=str(final_clip_path),
            )
            metadata_report.append(meta)
            
            # 5. Place on timeline
            start_ms = int(start_s * 1000)
            canvas = canvas.overlay(audio_segment, position=start_ms)
            
            # No explicit cleanup needed since we are using tempfile.TemporaryDirectory
                
        return canvas, metadata_report

    async def _generate_tts_with_retry(self, text: str, output_path: Path) -> None:
        """Generate TTS using edge-tts with bounded retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(str(output_path))
                
                if not output_path.exists():
                    raise SynthesisError("edge-tts finished but file was not created.")
                return
                
            except Exception as exc:
                logger.warning("TTS attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt >= self.max_retries:
                    raise SynthesisError(
                        f"Failed to generate TTS after {self.max_retries} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(2 ** attempt)

    def _apply_atempo(self, input_path: Path, output_path: Path, factor: float) -> None:
        """Use FFmpeg to speed up audio without changing pitch."""
        command = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-filter:a", f"atempo={factor}",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False
            )
        except FileNotFoundError as exc:
            raise SynthesisError("FFmpeg was not found on PATH.") from exc
            
        if result.returncode != 0:
            raise SynthesisError(f"FFmpeg atempo failed: {result.stderr}")
            
        if not output_path.exists():
            raise SynthesisError(f"FFmpeg output file not created: {output_path}")
