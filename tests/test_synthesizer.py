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
@pytest.mark.asyncio
async def test_process_all_segments_logic(
    mock_atempo, 
    mock_generate, 
    mock_from_file, 
    mock_synthesizer
):
    """
    Test speed factor logic, atempo thresholds, and timeline placement
    without actually calling edge-tts or ffmpeg.
    """
    segments = [
        # Segment 1: perfectly matches target duration (2.0s)
        {"id": 1, "start": 0.0, "end": 2.0, "translated_text": "A"},
        
        # Segment 2: slightly longer (generated 4.4s, target 4.0s) -> ratio 1.1 (within 1.05-1.25)
        {"id": 2, "start": 2.0, "end": 6.0, "translated_text": "B"},
        
        # Segment 3: way too long (generated 10.0s, target 4.0s) -> ratio 2.5 (> 1.25 limit)
        {"id": 3, "start": 6.0, "end": 10.0, "translated_text": "C"},
    ]
    
    # Mock duration returned by AudioSegment.from_file
    mock_audio = MagicMock()
    mock_audio.__len__.side_effect = [
        2000, # Seg 1: 2.0s
        4400, # Seg 2 raw: 4.4s
        10000, # Seg 3 raw: 10.0s
    ]
    mock_from_file.return_value = mock_audio
    
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
    assert report[0].generated_duration == 2.0
    assert report[0].speed_factor == 1.0
    assert report[0].applied_speed_factor == 1.0
    assert not report[0].timing_violation
    
    # Check Seg 2: Normal stretch (4.4 / 4.0 = 1.1)
    assert report[1].id == 2
    assert report[1].target_duration == 4.0
    assert report[1].generated_duration == 4.4
    assert report[1].speed_factor == 1.1
    assert report[1].applied_speed_factor == 1.1
    assert not report[1].timing_violation
    
    # Check Seg 3: Violation (10.0 / 4.0 = 2.5) -> cap at 1.25
    assert report[2].id == 3
    assert report[2].target_duration == 4.0
    assert report[2].generated_duration == 10.0
    assert report[2].speed_factor == 2.5
    assert report[2].applied_speed_factor == 1.25
    assert report[2].timing_violation
    
    # Check timeline placement
    assert mock_canvas.overlay.call_count == 3
    # Overlay is called with position in ms: 0, 2000, 6000
    mock_canvas.overlay.assert_any_call(mock_audio, position=0)
    mock_canvas.overlay.assert_any_call(mock_audio, position=2000)
    mock_canvas.overlay.assert_any_call(mock_audio, position=6000)


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
