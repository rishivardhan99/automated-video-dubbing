import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from pydub import AudioSegment

from src.synthesizer import (
    EdgeTTSSynthesizer,
    SynthesisError,
    SegmentMetadata,
)


@pytest.fixture
def mock_synthesizer():
    return EdgeTTSSynthesizer(voice="en-US-TestVoice", max_retries=1)


@patch("src.synthesizer.AudioSegment.from_file")
@patch.object(EdgeTTSSynthesizer, "_generate_tts_with_retry", new_callable=AsyncMock)
@patch.object(EdgeTTSSynthesizer, "_apply_atempo")
@patch.object(EdgeTTSSynthesizer, "_trim_silence")
@pytest.mark.asyncio
async def test_process_all_segments_logic(
    mock_trim,
    mock_atempo, 
    mock_generate, 
    mock_from_file, 
    mock_synthesizer
):
    """
    Test speed factor logic, available duration, and truncation.
    """
    segments = [
        # Segment 1: perfectly matches target duration (2.0s), available duration is 2.0s
        {"id": 1, "start": 0.0, "end": 2.0, "translated_text": "A"},
        
        # Segment 2: slightly longer (target 4.0s). available is 4.0s.
        {"id": 2, "start": 2.0, "end": 6.0, "translated_text": "B"},
        
        # Segment 3: way too long. target 4.0s, available 4.0s.
        {"id": 3, "start": 6.0, "end": 10.0, "translated_text": "C"},
    ]
    
    mock_audio = MagicMock(spec=AudioSegment)
    mock_from_file.return_value = mock_audio
    
    # We mock _trim_silence to return a mock audio whose len() changes
    mock_trimmed_1 = MagicMock(spec=AudioSegment)
    mock_trimmed_1.__len__.return_value = 2000
    
    mock_trimmed_2 = MagicMock(spec=AudioSegment)
    mock_trimmed_2.__len__.return_value = 4400
    
    mock_trimmed_3 = MagicMock(spec=AudioSegment)
    mock_trimmed_3.__len__.return_value = 10000
    
    mock_trim.side_effect = [mock_trimmed_1, mock_trimmed_2, mock_trimmed_3]
    
    # When atempo is called, from_file is called again. We need to handle its lengths.
    # Seg 2 stretched will be 4.0s (4000). Seg 3 stretched will be 10/1.25 = 8.0s (8000).
    mock_stretched_2 = MagicMock(spec=AudioSegment)
    mock_stretched_2.__len__.return_value = 4000
    mock_stretched_3 = MagicMock(spec=AudioSegment)
    mock_stretched_3.__len__.return_value = 8000
    
    # from_file is called for raw, then stretched. 
    # Seg1: raw
    # Seg2: raw, stretched
    # Seg3: raw, stretched
    mock_from_file.side_effect = [
        mock_audio, # Seg1 raw
        mock_audio, # Seg2 raw
        mock_stretched_2, # Seg2 stretched
        mock_audio, # Seg3 raw
        mock_stretched_3, # Seg3 stretched
    ]
    
    # Mock fade_out
    mock_faded_3 = MagicMock(spec=AudioSegment)
    mock_sliced_3 = MagicMock(spec=AudioSegment)
    mock_sliced_3.fade_out.return_value = mock_faded_3
    mock_stretched_3.__getitem__.return_value = mock_sliced_3
    
    mock_canvas = MagicMock(spec=AudioSegment)
    mock_canvas.overlay.return_value = mock_canvas

    output_dir = Path("/tmp/test_audio")
    
    canvas, report = await mock_synthesizer._process_all_segments(
        segments,
        mock_canvas,
        "job",
        output_dir
    )
    
    assert len(report) == 3
    
    # Check Seg 1: Perfect match
    assert report[0].id == 1
    assert report[0].target_duration == 2.0
    assert report[0].trimmed_tts_duration == 2.0
    assert report[0].required_speed_factor == 1.0
    assert report[0].applied_speed_factor == 1.0
    assert report[0].available_duration == 2.0
    assert not report[0].timing_violation
    assert not report[0].truncated
    
    # Check Seg 2: Normal stretch (4.4 / 4.0 = 1.1)
    assert report[1].id == 2
    assert report[1].target_duration == 4.0
    assert report[1].trimmed_tts_duration == 4.4
    assert report[1].required_speed_factor == 1.1
    assert report[1].applied_speed_factor == 1.1
    assert report[1].available_duration == 4.0
    assert not report[1].timing_violation
    assert not report[1].truncated
    
    # Check Seg 3: Violation (10.0 / 4.0 = 2.5) -> cap at 1.25. 
    # Final is 8.0s, available is 4.0s. Overflow is 4000ms.
    assert report[2].id == 3
    assert report[2].target_duration == 4.0
    assert report[2].trimmed_tts_duration == 10.0
    assert report[2].required_speed_factor == 2.5
    assert report[2].applied_speed_factor == 1.25
    assert report[2].available_duration == 4.0
    assert report[2].overflow_ms == 4000
    assert report[2].timing_violation
    assert report[2].truncated
    
    # Check timeline placement
    assert mock_canvas.overlay.call_count == 3
    mock_canvas.overlay.assert_any_call(mock_trimmed_1, position=0)
    mock_canvas.overlay.assert_any_call(mock_stretched_2, position=2000)
    mock_canvas.overlay.assert_any_call(mock_faded_3, position=6000)
    
    # Verify fade_out was called with 30ms
    mock_sliced_3.fade_out.assert_called_once_with(30)


@patch("src.synthesizer.get_audio_duration")
@patch("src.synthesizer.Path.exists")
@patch("src.synthesizer.Path.open")
@patch("src.synthesizer.asyncio.run")
@patch("src.synthesizer.AudioSegment.silent")
def test_synthesize_canvas_duration(
    mock_silent,
    mock_async_run,
    mock_open,
    mock_exists,
    mock_get_duration,
    mock_synthesizer
):
    """
    Test that the output timeline matches the exact source audio duration.
    """
    mock_exists.return_value = True
    # Simulate a 1-minute source video
    mock_get_duration.return_value = 60.03
    
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_file.read.return_value = '{"segments": [{"id": 0, "start": 0, "end": 1, "translated_text": "A"}]}'
    
    mock_canvas_result = MagicMock()
    mock_async_run.return_value = (mock_canvas_result, [])
    
    mock_synthesizer.synthesize(
        Path("dummy_trans.json"),
        Path("dummy_source.wav"),
    )
    
    # Check that AudioSegment.silent was initialized with exact duration
    # 60.03s -> 60030ms
    mock_silent.assert_called_once_with(duration=60030)
    mock_canvas_result.export.assert_called_once()

@patch("pydub.silence.detect_nonsilent")
def test_trim_silence(mock_detect, mock_synthesizer):
    """Test that silence is correctly trimmed with safety padding."""
    mock_audio = MagicMock(spec=AudioSegment)
    mock_audio.__len__.return_value = 10000
    
    # Non-silent speech from 1000ms to 5000ms
    mock_detect.return_value = [[1000, 2000], [2500, 5000]]
    
    mock_synthesizer.safety_pad = 50
    
    mock_synthesizer._trim_silence(mock_audio)
    
    # Should slice from (1000 - 50) to (5000 + 50)
    mock_audio.__getitem__.assert_called_once_with(slice(950, 5050))

