"""
Speaker diarization module using pyannote.audio.

This module is ONLY active when SPEAKER_MODE=diarized.
All heavy dependencies (torch, pyannote) are lazy-imported
so the core pipeline never loads them.
"""

import os
import time
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiarizationError(Exception):
    """Raised when speaker diarization fails."""


class PyAnnoteDiarizer:
    """
    Speaker diarizer using pyannote.audio.

    Heavy dependencies (torch, pyannote) are imported lazily
    inside methods — never at module level — so the core
    single-speaker pipeline is unaffected.
    """

    def __init__(self, hf_token: str | None = None) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise DiarizationError(
                "HF_TOKEN is required for speaker diarization. "
                "Create a free HuggingFace account, accept the "
                "pyannote model license at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1 "
                "and add HF_TOKEN=hf_xxxxx to your .env file."
            )

        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the pyannote pipeline on first use."""
        if self._pipeline is not None:
            return

        try:
            # Patch torchaudio to prevent AttributeError in newer versions
            import torchaudio
            if not hasattr(torchaudio, "AudioMetaData"):
                class _DummyMetaData:
                    pass
                torchaudio.AudioMetaData = _DummyMetaData
                if not hasattr(torchaudio, "info"):
                    torchaudio.info = lambda *args, **kwargs: _DummyMetaData()
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]

            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DiarizationError(
                "pyannote.audio is not installed. "
                "Install diarization dependencies with:\n"
                "  pip install -r requirements-diarization.txt\n"
                "This is only required for SPEAKER_MODE=diarized."
            ) from exc

        logger.info("Loading pyannote speaker diarization pipeline...")
        t0 = time.time()

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
        except Exception as exc:
            raise DiarizationError(
                f"Failed to load pyannote pipeline: {exc}\n"
                "Ensure your HF_TOKEN is valid and you have "
                "accepted the model license on HuggingFace."
            ) from exc

        load_time = time.time() - t0
        logger.info("Pyannote pipeline loaded in %.1fs", load_time)

    def diarize(self, audio_path: Path) -> list[dict]:
        """
        Run speaker diarization on an audio file.

        Returns a list of speaker intervals:
        [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.3},
            {"speaker": "SPEAKER_01", "start": 5.3, "end": 12.1},
            ...
        ]
        """
        if not audio_path.exists():
            raise DiarizationError(
                f"Audio file not found: {audio_path}"
            )

        self._load_pipeline()

        # Bypass torchaudio/torchcodec file loading bugs on Windows
        # by using pydub to load the audio into a torch tensor explicitly.
        from pydub import AudioSegment
        import torch
        import numpy as np

        audio = AudioSegment.from_file(str(audio_path))
        # Pyannote expects 16kHz mono audio by default
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy array of floats [-1.0, 1.0]
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        # Pyannote pipeline accepts a dict with "waveform" (shape: Channels x Samples)
        waveform = torch.from_numpy(samples).unsqueeze(0)
        
        logger.info("Running speaker diarization on %s...", audio_path.name)
        t0 = time.time()

        diarization_result = self._pipeline({
            "waveform": waveform,
            "sample_rate": 16000
        })

        processing_time = time.time() - t0

        # Convert pyannote output to normalized intervals
        intervals = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            intervals.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })

        # Sort by start time
        intervals.sort(key=lambda x: x["start"])

        # Collect unique speakers
        unique_speakers = sorted(set(iv["speaker"] for iv in intervals))

        logger.info(
            "Diarization complete in %.1fs — %d intervals, %d speakers detected",
            processing_time, len(intervals), len(unique_speakers),
        )
        for sp in unique_speakers:
            count = sum(1 for iv in intervals if iv["speaker"] == sp)
            total_duration = sum(
                iv["end"] - iv["start"]
                for iv in intervals if iv["speaker"] == sp
            )
            logger.info(
                "  %s: %d intervals, %.1fs total",
                sp, count, total_duration,
            )

        return intervals

    @staticmethod
    def assign_speakers(
        segments: list[dict],
        speaker_intervals: list[dict],
    ) -> list[dict]:
        """
        Assign a speaker label to each transcript segment
        using dominant temporal overlap.

        Each segment receives:
          - "speaker": the dominant speaker ID
          - "speaker_confidence": ratio of dominant overlap
            to total segment duration (0.0 – 1.0)

        Logs a warning when assignment confidence is below 0.6.
        """
        labelled_segments = []

        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                # Zero-length segment — assign unknown
                new_seg = dict(seg)
                new_seg["speaker"] = "SPEAKER_UNKNOWN"
                new_seg["speaker_confidence"] = 0.0
                labelled_segments.append(new_seg)
                continue

            # Calculate overlap with each speaker
            speaker_overlaps: dict[str, float] = {}

            for iv in speaker_intervals:
                overlap_start = max(seg_start, iv["start"])
                overlap_end = min(seg_end, iv["end"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > 0:
                    speaker = iv["speaker"]
                    speaker_overlaps[speaker] = (
                        speaker_overlaps.get(speaker, 0.0) + overlap
                    )

            new_seg = dict(seg)

            if not speaker_overlaps:
                # No diarization coverage for this segment
                new_seg["speaker"] = "SPEAKER_UNKNOWN"
                new_seg["speaker_confidence"] = 0.0
                logger.debug(
                    "Segment %d (%.1fs–%.1fs): no diarization coverage",
                    seg.get("id", -1), seg_start, seg_end,
                )
            else:
                # Find the dominant speaker
                dominant_speaker = max(
                    speaker_overlaps,
                    key=speaker_overlaps.get,
                )
                dominant_overlap = speaker_overlaps[dominant_speaker]
                confidence = round(dominant_overlap / seg_duration, 3)

                new_seg["speaker"] = dominant_speaker
                new_seg["speaker_confidence"] = confidence

                # Check for ambiguity
                if len(speaker_overlaps) > 1 and confidence < 0.6:
                    runner_up = sorted(
                        speaker_overlaps.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[1]
                    logger.warning(
                        "Segment %d (%.1fs–%.1fs): ambiguous speaker — "
                        "%s (%.0f%%) vs %s (%.0f%%). "
                        "Assigning dominant: %s",
                        seg.get("id", -1), seg_start, seg_end,
                        dominant_speaker,
                        confidence * 100,
                        runner_up[0],
                        round(runner_up[1] / seg_duration * 100),
                        dominant_speaker,
                    )

            labelled_segments.append(new_seg)

        # Summary
        assigned_speakers = set(
            s["speaker"] for s in labelled_segments
            if s["speaker"] != "SPEAKER_UNKNOWN"
        )
        unknown_count = sum(
            1 for s in labelled_segments
            if s["speaker"] == "SPEAKER_UNKNOWN"
        )
        low_confidence = sum(
            1 for s in labelled_segments
            if s.get("speaker_confidence", 1.0) < 0.6
            and s["speaker"] != "SPEAKER_UNKNOWN"
        )

        logger.info(
            "Speaker assignment complete: %d segments, "
            "%d speakers, %d unknown, %d low-confidence",
            len(labelled_segments),
            len(assigned_speakers),
            unknown_count,
            low_confidence,
        )

        return labelled_segments
