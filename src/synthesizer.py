import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment

from src.audio import get_audio_duration
from src.utils.logger import get_logger
from src.utils.paths import AUDIO_DIR

load_dotenv()

logger = get_logger(__name__)


class SynthesisError(Exception):
    """Raised when audio synthesis fails."""


@dataclass
class SegmentMetadata:
    id: int
    source_start: float
    source_end: float
    target_duration: float
    raw_tts_duration: float
    trimmed_tts_duration: float
    required_speed_factor: float
    applied_speed_factor: float
    available_duration: float
    overflow_ms: int
    timing_violation: bool
    truncated: bool
    output_clip_path: str
    final_duration_s: float


class EdgeTTSSynthesizer:
    def __init__(
        self,
        voice: str | None = None,
        max_retries: int = 3,
        speaker_voice_map: dict[str, str] | None = None,
    ) -> None:
        self.voice = voice or os.getenv("EDGE_TTS_VOICE", "en-US-GuyNeural")
        self.speaker_voice_map = speaker_voice_map
        self.max_retries = max_retries
        self.silence_threshold = float(os.getenv("TTS_SILENCE_THRESHOLD_DB", "-40.0"))
        self.min_silence_len = int(os.getenv("TTS_MIN_SILENCE_MS", "80"))
        self.safety_pad = int(os.getenv("TTS_SAFETY_PAD_MS", "40"))
        
        self.ducking_enabled = os.getenv("BACKGROUND_DUCKING", "true").lower() == "true"
        self.ducking_db = float(os.getenv("BACKGROUND_DUCK_DB", "8.0"))
        
        try:
            self.max_concurrency = int(os.getenv("TTS_MAX_CONCURRENCY", "10"))
        except ValueError:
            self.max_concurrency = 10

    def _trim_silence(self, audio_segment: AudioSegment) -> AudioSegment:
        from pydub.silence import detect_nonsilent
        nonsilent_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_threshold
        )
        if not nonsilent_ranges:
            return audio_segment # no speech detected or very short

        start_ms = nonsilent_ranges[0][0]
        end_ms = nonsilent_ranges[-1][1]

        start_ms = max(0, start_ms - self.safety_pad)
        end_ms = min(len(audio_segment), end_ms + self.safety_pad)

        return audio_segment[start_ms:end_ms]

    def synthesize(
        self,
        translation_path: Path,
        source_audio_path: Path,
        bg_music_path: Path | None = None,
    ) -> tuple[Path, list[SegmentMetadata]]:
        """
        Generate TTS for all translated segments and assemble them
        onto a canvas (either silent or background music).
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
            
        # Pydub works in milliseconds
        if bg_music_path and bg_music_path.exists():
            logger.info("Initializing synthesis canvas with background music from %s", bg_music_path.name)
            canvas = AudioSegment.from_file(str(bg_music_path))
            # Just to be safe, if Demucs shortened it slightly, we can pad it or let it be.
            # Demucs should preserve exact length.
        else:
            logger.info("Initializing silent synthesis canvas of duration %.2fs", source_duration_s)
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
        
        logger.info(f"Starting concurrent TTS synthesis for {len(segments)} segments (Concurrency: {self.max_concurrency})")
        
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_segment(i: int, segment: dict):
            async with semaphore:
                seg_id = segment["id"]
                start_s = segment["start"]
                end_s = segment["end"]
                text = segment["translated_text"]
                
                target_duration_s = end_s - start_s
                
                # Pre-TTS sanity check
                est_duration_s = len(text) / 15.0
                if target_duration_s > 0 and (est_duration_s / target_duration_s) > 1.25:
                    logger.debug(
                        "Segment %d translation likely exceeds available duration (est %.1fs vs target %.1fs)",
                        seg_id, est_duration_s, target_duration_s
                    )

                # Available duration (to prevent overlaps, but allow natural length)
                if i + 1 < len(segments):
                    available_duration_s = max(0.1, segments[i + 1]["start"] - start_s)
                    target_duration_s = min(target_duration_s, available_duration_s)
                else:
                    available_duration_s = target_duration_s

                logger.debug(
                    "Synthesizing segment %d (%.2fs - %.2fs, target duration: %.2fs, available: %.2fs)", 
                    seg_id, start_s, end_s, target_duration_s, available_duration_s
                )
                
                raw_clip_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_raw.mp3"
                
                # Variables for metadata tracking
                raw_duration_s = 0.0
                trimmed_duration_s = 0.0
                required_speed_factor = 1.0
                applied_speed_factor = 1.0
                overflow_ms = 0
                violation = False
                truncated = False
                final_clip_path = raw_clip_path
                
                # 1. Generate TTS
                if not text.strip():
                    logger.debug("Segment %d text is empty. Creating silent placeholder.", seg_id)
                    audio_segment = AudioSegment.silent(duration=0)
                    raw_clip_path.touch()
                    final_duration_s = 0.0
                else:
                    # Determine voice for this segment
                    segment_voice = self.voice
                    if self.speaker_voice_map and "speaker" in segment:
                        segment_voice = self.speaker_voice_map.get(
                            segment["speaker"], self.voice
                        )

                    await self._generate_tts_with_retry(
                        text, raw_clip_path, voice=segment_voice,
                    )
                    
                    # Offload pydub I/O to a thread so we don't block asyncio loop
                    def process_audio():
                        raw_audio = AudioSegment.from_file(str(raw_clip_path))
                        return raw_audio
                    
                    raw_audio_segment = await asyncio.to_thread(process_audio)
                    raw_duration_s = len(raw_audio_segment) / 1000.0
                    
                    # Trim silence
                    audio_segment = await asyncio.to_thread(self._trim_silence, raw_audio_segment)
                    trimmed_duration_s = len(audio_segment) / 1000.0
                    removed_ms = len(raw_audio_segment) - len(audio_segment)
                    logger.debug("Trimmed %d ms silence from segment %d. Raw: %.2fs, Trimmed: %.2fs", 
                                removed_ms, seg_id, raw_duration_s, trimmed_duration_s)
                    
                    trimmed_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_trimmed.wav"
                    await asyncio.to_thread(audio_segment.export, str(trimmed_path), format="wav")
                    
                    if target_duration_s > 0:
                        required_speed_factor = trimmed_duration_s / target_duration_s

                    # Apply time-stretching logic
                    if required_speed_factor <= 1.05:
                        final_clip_path = trimmed_path
                    else:
                        if required_speed_factor <= 1.5:
                            applied_speed_factor = required_speed_factor
                        else:
                            applied_speed_factor = 1.5
                            violation = True
                            logger.debug(
                                "[TIMING VIOLATION] Segment %d generated %.2fs audio for %.2fs target. Max 1.5x stretch applied.",
                                seg_id, trimmed_duration_s, target_duration_s
                            )

                        final_clip_path = output_dir / f"{job_prefix}_seg{seg_id:03d}_stretched.wav"
                        await asyncio.to_thread(self._apply_atempo, trimmed_path, final_clip_path, applied_speed_factor)
                        
                        def load_final():
                            return AudioSegment.from_file(str(final_clip_path))
                        audio_segment = await asyncio.to_thread(load_final)

                    # Overlap Prevention & Defensive Cutoff
                    final_duration_s = len(audio_segment) / 1000.0
                    if final_duration_s > available_duration_s:
                        overflow_ms = int((final_duration_s - available_duration_s) * 1000)
                        logger.debug("Segment %d overflows available duration by %d ms. Truncating with fade-out.", seg_id, overflow_ms)
                        
                        available_ms = int(available_duration_s * 1000)
                        
                        def truncate_audio():
                            return audio_segment[:available_ms].fade_out(30)
                        audio_segment = await asyncio.to_thread(truncate_audio)
                        
                        truncated = True
                        violation = True
                        final_duration_s = len(audio_segment) / 1000.0
                
                logger.info(f"Generated Segment {seg_id} | Final length: {final_duration_s:.2f}s")
                
                # Record metadata
                meta = SegmentMetadata(
                    id=seg_id,
                    source_start=start_s,
                    source_end=end_s,
                    target_duration=target_duration_s,
                    raw_tts_duration=raw_duration_s,
                    trimmed_tts_duration=trimmed_duration_s,
                    required_speed_factor=required_speed_factor,
                    applied_speed_factor=applied_speed_factor,
                    available_duration=available_duration_s,
                    overflow_ms=overflow_ms,
                    timing_violation=violation,
                    truncated=truncated,
                    output_clip_path=str(final_clip_path),
                    final_duration_s=final_duration_s,
                )
                
                return (i, start_s, audio_segment, meta)

        # Create tasks and run concurrently
        tasks = [process_segment(i, segment) for i, segment in enumerate(segments)]
        results = await asyncio.gather(*tasks)
        
        # Sort by index to maintain chronology
        results.sort(key=lambda x: x[0])
        
        metadata_report = []
        
        logger.info("Concurrency phase complete! Assembling TTS segments onto canvas...")
        
        # Sequentially place segments onto the master canvas
        for _, start_s, audio_segment, meta in results:
            metadata_report.append(meta)
            
            start_ms = int(start_s * 1000)
            final_duration_s = meta.final_duration_s
            
            # Background ducking
            if self.ducking_enabled and final_duration_s > 0:
                end_ms = start_ms + int(final_duration_s * 1000)
                # Ensure we don't try to duck past the canvas length
                end_ms = min(end_ms, len(canvas))
                if start_ms < len(canvas):
                    before = canvas[:start_ms]
                    during = canvas[start_ms:end_ms] - self.ducking_db
                    after = canvas[end_ms:]
                    canvas = before + during + after
            
            # Overlay
            if final_duration_s > 0:
                canvas = canvas.overlay(audio_segment, position=start_ms)
                
        logger.info("Assembly complete!")
        return canvas, metadata_report

    async def _generate_tts_with_retry(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
    ) -> None:
        """Generate TTS using edge-tts with bounded retries."""
        effective_voice = voice or self.voice
        for attempt in range(1, self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(text, effective_voice)
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
