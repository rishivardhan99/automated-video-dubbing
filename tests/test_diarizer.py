"""
Tests for speaker diarization and voice mapping.

All tests are offline — no real pyannote, HuggingFace,
or Edge-TTS calls. Everything is mocked.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.diarizer import PyAnnoteDiarizer, DiarizationError
from src.speaker_voices import build_speaker_voice_map, DEFAULT_VOICE_POOL


# ===================================================
# DIARIZATION ERROR HANDLING
# ===================================================


def test_missing_hf_token_error():
    """Missing HF_TOKEN produces a clear actionable error."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(DiarizationError, match="HF_TOKEN"):
            PyAnnoteDiarizer(hf_token=None)


def test_missing_pyannote_import_error():
    """Missing pyannote raises clear install instructions."""
    diarizer = PyAnnoteDiarizer(hf_token="fake_token")

    with patch.dict("sys.modules", {"pyannote": None, "pyannote.audio": None}):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'pyannote'"),
        ):
            with pytest.raises(DiarizationError, match="pyannote.audio is not installed"):
                diarizer._load_pipeline()


# ===================================================
# SPEAKER ASSIGNMENT — DOMINANT OVERLAP
# ===================================================


def test_speaker_assignment_single_speaker():
    """All segments assigned to the single detected speaker."""
    segments = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "Hello"},
        {"id": 1, "start": 5.0, "end": 10.0, "text": "World"},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 12.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_00"
    # Original fields preserved
    assert result[0]["text"] == "Hello"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 5.0
    assert result[0]["id"] == 0


def test_speaker_assignment_two_speakers():
    """Correct dominant-overlap assignment with two speakers."""
    segments = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "First speaker"},
        {"id": 1, "start": 6.0, "end": 11.0, "text": "Second speaker"},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.5},
        {"speaker": "SPEAKER_01", "start": 5.5, "end": 12.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_speaker_assignment_ambiguous_overlap():
    """
    When a segment overlaps two speakers substantially,
    the dominant speaker is assigned and confidence is low.
    """
    # Segment spans 0.0–10.0
    # SPEAKER_00 covers 0.0–5.5 (5.5s overlap = 55%)
    # SPEAKER_01 covers 5.0–10.0 (5.0s overlap = 50%, but only 4.5s exclusive)
    # Actually: overlap with SPEAKER_00 = 5.5s, SPEAKER_01 = 5.0s
    segments = [
        {"id": 0, "start": 0.0, "end": 10.0, "text": "Ambiguous segment"},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.5},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["speaker"] == "SPEAKER_00"
    # Confidence should be moderate (0.55)
    assert result[0]["speaker_confidence"] < 0.6


def test_speaker_assignment_no_coverage():
    """Segments with no diarization coverage get SPEAKER_UNKNOWN."""
    segments = [
        {"id": 0, "start": 100.0, "end": 105.0, "text": "No coverage"},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["speaker"] == "SPEAKER_UNKNOWN"
    assert result[0]["speaker_confidence"] == 0.0


def test_speaker_assignment_zero_duration():
    """Zero-duration segments get SPEAKER_UNKNOWN."""
    segments = [
        {"id": 0, "start": 5.0, "end": 5.0, "text": ""},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["speaker"] == "SPEAKER_UNKNOWN"
    assert result[0]["speaker_confidence"] == 0.0


def test_speaker_assignment_preserves_fields():
    """
    Speaker assignment must not alter text, start, end, or id.
    Only adds speaker and speaker_confidence.
    """
    segments = [
        {"id": 42, "start": 12.5, "end": 17.3, "text": "Preserved text"},
    ]
    intervals = [
        {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["id"] == 42
    assert result[0]["start"] == 12.5
    assert result[0]["end"] == 17.3
    assert result[0]["text"] == "Preserved text"
    assert result[0]["speaker"] == "SPEAKER_01"
    assert "speaker_confidence" in result[0]


def test_speaker_assignment_high_confidence():
    """Full coverage by a single speaker yields confidence ~1.0."""
    segments = [
        {"id": 0, "start": 2.0, "end": 8.0, "text": "Full coverage"},
    ]
    intervals = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
    ]

    result = PyAnnoteDiarizer.assign_speakers(segments, intervals)

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["speaker_confidence"] == 1.0


# ===================================================
# SPEAKER VOICE MAPPING
# ===================================================


def test_speaker_voice_map_defaults():
    """Speakers get round-robin voices from the default pool."""
    speakers = {"SPEAKER_00", "SPEAKER_01"}

    with patch.dict("os.environ", {}, clear=True):
        voice_map = build_speaker_voice_map(speakers)

    assert len(voice_map) == 2
    assert voice_map["SPEAKER_00"] != voice_map["SPEAKER_01"]
    # All assigned voices come from the pool
    for voice in voice_map.values():
        assert voice in DEFAULT_VOICE_POOL


def test_speaker_voice_map_env_override():
    """Explicit env-var overrides take priority."""
    speakers = {"SPEAKER_00", "SPEAKER_01"}

    with patch.dict("os.environ", {
        "SPEAKER_00_VOICE": "en-US-CustomVoice",
    }, clear=True):
        voice_map = build_speaker_voice_map(speakers)

    assert voice_map["SPEAKER_00"] == "en-US-CustomVoice"
    # SPEAKER_01 still gets a pool voice
    assert voice_map["SPEAKER_01"] in DEFAULT_VOICE_POOL


def test_speaker_voice_map_deterministic():
    """Same input always produces the same output."""
    speakers = {"SPEAKER_02", "SPEAKER_00", "SPEAKER_01"}

    with patch.dict("os.environ", {}, clear=True):
        map1 = build_speaker_voice_map(speakers)
        map2 = build_speaker_voice_map(speakers)

    assert map1 == map2


def test_speaker_voice_map_more_speakers_than_voices():
    """When speakers exceed pool size, voices wrap around."""
    # Create more speakers than voices in the pool
    speakers = {f"SPEAKER_{i:02d}" for i in range(10)}

    with patch.dict("os.environ", {}, clear=True):
        voice_map = build_speaker_voice_map(speakers)

    assert len(voice_map) == 10
    # All assigned voices are from the pool (wrapping)
    for voice in voice_map.values():
        assert voice in DEFAULT_VOICE_POOL


def test_speaker_voice_map_with_default_voice():
    """Default voice is placed first in the pool."""
    speakers = {"SPEAKER_00"}

    with patch.dict("os.environ", {}, clear=True):
        voice_map = build_speaker_voice_map(
            speakers, default_voice="en-US-SteffanNeural",
        )

    # SPEAKER_00 (first sorted speaker) should get the default voice
    assert voice_map["SPEAKER_00"] == "en-US-SteffanNeural"


# ===================================================
# SYNTHESIZER — SPEAKER-AWARE VOICE SELECTION
# ===================================================


def test_single_mode_no_speaker_key():
    """
    Without speaker_voice_map, synthesizer uses self.voice
    regardless of segment contents.
    """
    from src.synthesizer import EdgeTTSSynthesizer

    synth = EdgeTTSSynthesizer(
        voice="en-US-TestVoice",
        speaker_voice_map=None,
    )

    # No speaker_voice_map → always self.voice
    assert synth.voice == "en-US-TestVoice"
    assert synth.speaker_voice_map is None


def test_diarized_mode_voice_map_stored():
    """
    With speaker_voice_map, synthesizer stores the mapping.
    """
    from src.synthesizer import EdgeTTSSynthesizer

    voice_map = {
        "SPEAKER_00": "en-US-SteffanNeural",
        "SPEAKER_01": "en-US-AriaNeural",
    }

    synth = EdgeTTSSynthesizer(
        voice="en-US-DefaultVoice",
        speaker_voice_map=voice_map,
    )

    assert synth.speaker_voice_map == voice_map
    # Default voice is still set as fallback
    assert synth.voice == "en-US-DefaultVoice"


# ===================================================
# TRANSLATOR — SPEAKER METADATA PRESERVATION
# ===================================================


def test_translator_preserves_speaker_metadata():
    """
    When segments have speaker/speaker_confidence keys,
    the English-skip path preserves them.
    """
    import json
    import tempfile

    from src.translator import GroqTranslator

    transcript = {
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
                "speaker": "SPEAKER_00",
                "speaker_confidence": 0.95,
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 10.0,
                "text": "Goodbye world",
                "speaker": "SPEAKER_01",
                "speaker_confidence": 0.88,
            },
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as f:
        json.dump(transcript, f, ensure_ascii=False)
        tmp_path = Path(f.name)

    try:
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}):
            translator = GroqTranslator()
            result_path = translator.translate(tmp_path)

            with result_path.open("r", encoding="utf-8") as rf:
                result = json.load(rf)

            segs = result["segments"]
            assert segs[0]["speaker"] == "SPEAKER_00"
            assert segs[0]["speaker_confidence"] == 0.95
            assert segs[1]["speaker"] == "SPEAKER_01"
            assert segs[1]["speaker_confidence"] == 0.88
    finally:
        tmp_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def test_translator_no_speaker_metadata():
    """
    When segments have no speaker keys (single mode),
    translated output also has no speaker keys.
    """
    import json
    import tempfile

    from src.translator import GroqTranslator

    transcript = {
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as f:
        json.dump(transcript, f, ensure_ascii=False)
        tmp_path = Path(f.name)

    try:
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake_key"}):
            translator = GroqTranslator()
            result_path = translator.translate(tmp_path)

            with result_path.open("r", encoding="utf-8") as rf:
                result = json.load(rf)

            segs = result["segments"]
            assert "speaker" not in segs[0]
            assert "speaker_confidence" not in segs[0]
    finally:
        tmp_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
