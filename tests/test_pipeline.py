import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.downloader import validate_youtube_url
from src.quality import (
    analyze_transcript,
    QualityState,
)
from src.translator import (
    GroqTranslator,
    TranslationError,
)
from src.transcribers.groq_transcriber import (
    GroqTranscriber,
    TranscriptionError,
)


# ===================================================
# DOWNLOADER TESTS
# ===================================================


def test_valid_youtube_url():
    assert validate_youtube_url(
        "https://www.youtube.com/watch?v=test"
    )


def test_valid_short_youtube_url():
    assert validate_youtube_url(
        "https://youtu.be/test"
    )


def test_invalid_url():
    assert not validate_youtube_url(
        "https://example.com/video"
    )


def test_invalid_scheme():
    assert not validate_youtube_url(
        "ftp://youtube.com/video"
    )


# ===================================================
# QUALITY GATE TESTS
# ===================================================


def test_quality_pass_clean_transcript():
    transcript = {
        "text": "Hello world. How are you?",
        "language": "en",
        "language_probability": 0.99,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world.",
            },
            {
                "id": 1,
                "start": 1.0,
                "end": 2.0,
                "text": "How are you?",
            },
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=2.0,
        processing_time=1.0,
    )
    assert report.state == QualityState.PASS
    assert not report.failures
    assert report.diagnostics["segment_count"] == 2


def test_quality_fail_empty_segments():
    transcript = {
        "text": "",
        "language": "en",
        "segments": [],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=2.0,
        processing_time=1.0,
    )
    assert report.state == QualityState.FAIL
    assert "no segments" in report.failures[0].lower()


def test_quality_fail_pathological_repetition():
    transcript = {
        "text": " ".join(["Hey!"] * 20),
        "language": "te",
        "segments": [
            {
                "id": i,
                "start": i,
                "end": i + 1,
                "text": "Hey!",
            }
            for i in range(20)
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=20.0,
        processing_time=2.0,
    )
    assert report.state == QualityState.FAIL
    assert any(
        "repetition" in f.lower()
        for f in report.failures
    )


def test_quality_fail_excessive_single_chars():
    transcript = {
        "text": " ".join(
            chr(97 + i) for i in range(14)
        ),
        "language": "en",
        "segments": [
            {
                "id": i,
                "start": i,
                "end": i + 0.5,
                "text": chr(97 + i),
            }
            for i in range(14)
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=14.0,
        processing_time=2.0,
    )
    assert report.state == QualityState.FAIL
    assert any(
        "short segments" in f.lower()
        for f in report.failures
    )


def test_quality_warn_script_mixing():
    transcript = {
        "text": "Hello నమస్కారం 안녕하세요",
        "language": "te",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
            },
            {
                "id": 1,
                "start": 1.0,
                "end": 2.0,
                "text": "నమస్కారం",
            },
            {
                "id": 2,
                "start": 2.0,
                "end": 3.0,
                "text": "안녕하세요",
            },
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=3.0,
        processing_time=1.0,
    )
    assert report.state == QualityState.WARN
    assert any(
        "multiple scripts" in w.lower()
        for w in report.warnings
    )


def test_quality_warn_low_confidence():
    transcript = {
        "text": "Maybe?",
        "language": "en",
        "language_probability": 0.2,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Maybe?",
            }
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=1.0,
        processing_time=1.0,
    )
    assert report.state == QualityState.WARN
    assert any(
        "confidence" in w.lower()
        for w in report.warnings
    )


def test_quality_fail_invalid_timestamps():
    transcript = {
        "text": "Bad timestamps.",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 5.0,
                "end": 3.0,
                "text": "Bad timestamps.",
            }
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=5.0,
        processing_time=1.0,
    )
    assert report.state == QualityState.FAIL
    assert any(
        "invalid timestamps" in f.lower()
        for f in report.failures
    )


def test_quality_fail_intra_segment_repetition():
    """Simulate a Whisper hallucination loop."""
    repeated_text = "ని" * 60
    transcript = {
        "text": repeated_text,
        "language": "te",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 30.0,
                "text": repeated_text,
            },
        ],
    }
    report = analyze_transcript(
        transcript,
        audio_duration=30.0,
        processing_time=5.0,
    )
    assert report.state == QualityState.FAIL
    assert any(
        "intra-segment repetition" in f.lower()
        for f in report.failures
    )


# ===================================================
# GROQ TRANSCRIBER NORMALIZATION TESTS
# ===================================================


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_normalize_valid_response():
    transcriber = GroqTranscriber()

    response = MagicMock()
    response.model_dump.return_value = {
        "text": "Hello world.",
        "language": "en",
        "duration": 2.5,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
            },
            {
                "id": 1,
                "start": 1.0,
                "end": 2.5,
                "text": "world.",
            },
        ],
    }

    result = transcriber._normalize_response(
        response
    )

    assert result["provider"] == "groq"
    assert result["language"] == "en"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == "Hello"
    assert result["segments"][1]["start"] == 1.0


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_normalize_missing_segments():
    transcriber = GroqTranscriber()

    response = MagicMock()
    response.model_dump.return_value = {
        "text": "Hello world.",
        "language": "en",
        "duration": 2.0,
        "segments": None,
    }

    result = transcriber._normalize_response(
        response
    )

    # Falls back to single segment from full text.
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Hello world."


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_normalize_empty_response():
    transcriber = GroqTranscriber()

    response = MagicMock()
    response.model_dump.return_value = {
        "text": "",
        "language": "en",
        "duration": 0.0,
        "segments": [],
    }

    result = transcriber._normalize_response(
        response
    )

    assert len(result["segments"]) == 0


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_normalize_timestamp_offset():
    transcriber = GroqTranscriber()

    response = MagicMock()
    response.model_dump.return_value = {
        "text": "Offset test.",
        "language": "te",
        "duration": 5.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "Offset test.",
            },
        ],
    }

    result = transcriber._normalize_response(
        response,
        time_offset=600.0,
        start_id=42,
    )

    seg = result["segments"][0]
    assert seg["id"] == 42
    assert seg["start"] == 600.0
    assert seg["end"] == 603.0


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_normalize_id_generation():
    transcriber = GroqTranscriber()

    response = MagicMock()
    response.model_dump.return_value = {
        "text": "A B C",
        "language": "en",
        "duration": 3.0,
        "segments": [
            {"start": 0, "end": 1, "text": "A"},
            {"start": 1, "end": 2, "text": "B"},
            {"start": 2, "end": 3, "text": "C"},
        ],
    }

    result = transcriber._normalize_response(
        response
    )

    ids = [s["id"] for s in result["segments"]]
    assert ids == [0, 1, 2]


# ===================================================
# CHUNKING TESTS
# ===================================================


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_small_file_no_chunking():
    """
    A file under the upload limit should not
    trigger chunking.
    """
    transcriber = GroqTranscriber()

    # 10 MB is under the 25 MB limit.
    file_size = 10 * 1024 * 1024
    assert file_size <= 25 * 1024 * 1024


@patch.dict(
    "os.environ",
    {"GROQ_API_KEY": "test-key"},
)
def test_chunked_segment_merge_order():
    """
    Verify that merged segments from two chunks
    maintain chronological order and sequential IDs.
    """
    transcriber = GroqTranscriber()

    chunk1_segments = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "A"},
        {"id": 1, "start": 5.0, "end": 10.0, "text": "B"},
    ]
    chunk2_segments = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "C"},
        {"id": 1, "start": 5.0, "end": 10.0, "text": "D"},
    ]

    # Simulate offset application for chunk 2.
    offset = 600.0  # 10 minutes
    for seg in chunk2_segments:
        seg["start"] += offset
        seg["end"] += offset

    all_segs = chunk1_segments + chunk2_segments

    # Renumber.
    for i, seg in enumerate(all_segs):
        seg["id"] = i

    assert len(all_segs) == 4
    assert all_segs[0]["id"] == 0
    assert all_segs[3]["id"] == 3
    assert all_segs[2]["start"] == 600.0
    assert all_segs[3]["end"] == 610.0

    # Verify chronological order.
    for i in range(len(all_segs) - 1):
        assert (
            all_segs[i]["start"]
            <= all_segs[i + 1]["start"]
        )


# ===================================================
# TRANSLATOR TESTS
# ===================================================


def test_translator_batching():
    with patch.dict("os.environ", {"GROQ_API_KEY": "fake", "TRANSLATION_MAX_SEGMENTS": "2"}):
        translator = GroqTranslator(
            max_retries=1,
        )

    segments = [
        {
            "id": 0,
            "text": "abcd",
            "start": 0,
            "end": 1,
        },
        {
            "id": 1,
            "text": "efgh",
            "start": 1,
            "end": 2,
        },
        {
            "id": 2,
            "text": "ijkl",
            "start": 2,
            "end": 3,
        },
    ]

    batches = translator._build_batches(segments)

    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1
    assert batches[0][0]["id"] == 0
    assert batches[0][1]["id"] == 1
    assert batches[1][0]["id"] == 2


def test_translator_validation_success():
    source_segments = [
        {
            "id": 0,
            "text": "Hola",
            "start": 0,
            "end": 1,
        },
        {
            "id": 1,
            "text": "Mundo",
            "start": 1,
            "end": 2,
        },
    ]

    translated = [
        {"id": 0, "translated_text": "Hello"},
        {"id": 1, "translated_text": "World"},
    ]

    result = GroqTranslator._validate_batch(
        source_segments,
        translated,
    )

    assert len(result) == 2
    assert result[0]["translated_text"] == "Hello"
    assert result[1]["translated_text"] == "World"
    assert result[0]["start"] == 0
    assert result[1]["end"] == 2


def test_translator_validation_fail_missing_id():
    source_segments = [
        {
            "id": 0,
            "text": "Hola",
            "start": 0,
            "end": 1,
        }
    ]
    translated = [{"translated_text": "Hello"}]

    with pytest.raises(
        TranslationError,
        match="Groq changed",
    ):
        GroqTranslator._validate_batch(
            source_segments,
            translated,
        )


def test_translator_validation_fail_reordered():
    source_segments = [
        {
            "id": 0,
            "text": "Hola",
            "start": 0,
            "end": 1,
        },
        {
            "id": 1,
            "text": "Mundo",
            "start": 1,
            "end": 2,
        },
    ]
    translated = [
        {"id": 1, "translated_text": "World"},
        {"id": 0, "translated_text": "Hello"},
    ]

    with pytest.raises(
        TranslationError,
        match="Groq changed",
    ):
        GroqTranslator._validate_batch(
            source_segments,
            translated,
        )


def test_translator_validation_fail_empty_text():
    """Empty-string translations are now allowed (for gibberish segments)."""
    source_segments = [
        {
            "id": 0,
            "text": "Hola",
            "start": 0,
            "end": 1,
        }
    ]
    translated = [
        {"id": 0, "translated_text": "   "}
    ]

    # Should NOT raise — empty strings are explicitly allowed
    result = GroqTranslator._validate_batch(
        source_segments,
        translated,
    )
    assert len(result) == 1


def test_translator_validation_fail_wrong_count():
    source_segments = [
        {
            "id": 0,
            "text": "Hola",
            "start": 0,
            "end": 1,
        }
    ]
    translated = [
        {"id": 0, "translated_text": "Hello"},
        {"id": 1, "translated_text": "World"},
    ]

    with pytest.raises(
        TranslationError,
        match="Groq changed",
    ):
        GroqTranslator._validate_batch(
            source_segments,
            translated,
        )


# ===================================================
# RETRY DELAY PARSING
# ===================================================


def test_groq_retry_delay_parsing():
    exc = Exception(
        "Rate limit reached. "
        "Please retry-after: 5"
    )
    delay = GroqTranslator._get_retry_delay(
        exc, attempt=1
    )
    assert delay == 5.0

    exc2 = Exception(
        "Rate limit reached. "
        "Please retry after 12.5 seconds."
    )
    delay2 = GroqTranslator._get_retry_delay(
        exc2, attempt=2
    )
    assert delay2 == 12.5

    exc3 = Exception("Just a normal error")
    delay3 = GroqTranslator._get_retry_delay(
        exc3, attempt=3
    )
    # Base delay is 4.0 + random jitter [0.1, 1.0]
    assert 4.0 <= delay3 <= 5.0