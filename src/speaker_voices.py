"""
Deterministic speaker-to-voice mapping for multi-speaker dubbing.

Maps detected speaker IDs (e.g. SPEAKER_00, SPEAKER_01) to
distinct Edge-TTS voices. Supports explicit env-var overrides
and falls back to a curated voice pool with round-robin assignment.
"""

import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Curated pool of high-quality, distinct Edge-TTS voices.
# Ordered to maximise auditory contrast between adjacent speakers.
DEFAULT_VOICE_POOL = [
    "en-US-SteffanNeural",       # Male, clear & articulate
    "en-US-AriaNeural",          # Female, warm & natural
    "en-US-ChristopherNeural",   # Male, deep & authoritative
    "en-US-JennyNeural",         # Female, friendly
    "en-GB-RyanNeural",          # Male, British accent
    "en-US-MichelleNeural",      # Female, professional
]


def build_speaker_voice_map(
    speakers: set[str],
    default_voice: str | None = None,
) -> dict[str, str]:
    """
    Build a deterministic mapping from speaker IDs to Edge-TTS voices.

    Priority:
      1. Explicit env-var override: SPEAKER_00_VOICE=en-US-AriaNeural
      2. Round-robin from DEFAULT_VOICE_POOL (sorted speaker order)

    Args:
        speakers: Set of speaker IDs detected by diarization.
        default_voice: The configured default EDGE_TTS_VOICE. Used as
                       the first voice in the pool if provided, ensuring
                       the primary speaker sounds consistent with
                       single-speaker mode.

    Returns:
        Dict mapping speaker ID → Edge-TTS voice name.
    """
    # Build the effective voice pool
    if default_voice and default_voice not in DEFAULT_VOICE_POOL:
        voice_pool = [default_voice] + DEFAULT_VOICE_POOL
    elif default_voice:
        # Move default_voice to the front
        pool = list(DEFAULT_VOICE_POOL)
        pool.remove(default_voice)
        voice_pool = [default_voice] + pool
    else:
        voice_pool = list(DEFAULT_VOICE_POOL)

    # Sort speakers for deterministic assignment
    sorted_speakers = sorted(speakers)

    voice_map: dict[str, str] = {}

    for i, speaker in enumerate(sorted_speakers):
        # Check for explicit env-var override
        # e.g. SPEAKER_00_VOICE=en-US-AriaNeural
        env_key = f"{speaker.replace(' ', '_')}_VOICE"
        env_override = os.getenv(env_key)

        if env_override:
            voice_map[speaker] = env_override
            logger.info(
                "  %s → %s (env override: %s)",
                speaker, env_override, env_key,
            )
        else:
            # Round-robin from the pool
            voice = voice_pool[i % len(voice_pool)]
            voice_map[speaker] = voice
            logger.info("  %s → %s", speaker, voice)

    return voice_map
