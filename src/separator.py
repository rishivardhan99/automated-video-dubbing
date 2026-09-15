import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SeparationError(Exception):
    """Raised when audio separation fails."""


@dataclass
class SeparationResult:
    vocals_path: Path
    background_path: Path
    provider: str
    processing_time: float


class AudioSeparator:
    def __init__(self, mode: str = "none"):
        self.mode = mode.lower()

    def separate(self, audio_path: Path) -> Optional[SeparationResult]:
        if self.mode == "none":
            logger.info("Audio separation disabled (AUDIO_SEPARATION=none)")
            return None

        if self.mode != "demucs":
            logger.warning("Unknown separation mode: %s. Falling back to none.", self.mode)
            return None

        if not audio_path.exists():
            raise SeparationError(f"Audio file not found: {audio_path}")

        return self._run_demucs(audio_path)

    def _run_demucs(self, audio_path: Path) -> SeparationResult:
        isolated_dir = Path("data/audio/isolated")
        isolated_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running Demucs vocal separation on %s...", audio_path.name)
        t0 = time.time()

        # Windows encoding fix for demucs printing Unicode characters to stdout
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            # Invoke using sys.executable to ensure we use the current virtual environment's Demucs
            command = [
                sys.executable,
                "-m", "demucs",
                "--two-stems=vocals",
                "-o", str(isolated_dir),
                str(audio_path)
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                encoding="utf-8",
                check=False
            )
        except Exception as exc:
            raise SeparationError(f"Failed to execute Demucs: {exc}") from exc

        if result.returncode != 0:
            logger.error("Demucs stdout: %s", result.stdout)
            raise SeparationError(f"Demucs failed:\n{result.stderr}")

        processing_time = time.time() - t0

        # Demucs outputs to <output_dir>/<model_name>/<filename>/
        # Default model is htdemucs
        vocals_path = isolated_dir / "htdemucs" / audio_path.stem / "vocals.wav"
        bg_path = isolated_dir / "htdemucs" / audio_path.stem / "no_vocals.wav"

        if not vocals_path.exists() or not bg_path.exists():
            raise SeparationError("Demucs succeeded but expected output files are missing.")

        logger.info("Demucs separation complete in %.2fs", processing_time)

        return SeparationResult(
            vocals_path=vocals_path,
            background_path=bg_path,
            provider="demucs",
            processing_time=processing_time
        )
