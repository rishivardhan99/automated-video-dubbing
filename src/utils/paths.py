from pathlib import Path


# Project root:
# automated-video-dubbing/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSLATIONS_DIR = DATA_DIR / "translations"
OUTPUT_DIR = DATA_DIR / "output"


def ensure_directories() -> None:
    """Create all required data directories if they don't exist."""
    for directory in (
        INPUT_DIR,
        AUDIO_DIR,
        TRANSCRIPTS_DIR,
        TRANSLATIONS_DIR,
        OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)