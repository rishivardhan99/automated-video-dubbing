import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.separator import AudioSeparator, SeparationError

def test_separator_none_mode():
    separator = AudioSeparator(mode="none")
    result = separator.separate(Path("dummy.wav"))
    assert result is None

def test_separator_unknown_mode():
    separator = AudioSeparator(mode="invalid")
    result = separator.separate(Path("dummy.wav"))
    assert result is None

@patch("src.separator.Path.exists")
def test_separator_missing_file(mock_exists):
    mock_exists.return_value = False
    separator = AudioSeparator(mode="demucs")
    with pytest.raises(SeparationError, match="Audio file not found"):
        separator.separate(Path("missing.wav"))

@patch("src.separator.subprocess.run")
@patch("src.separator.Path.exists")
@patch("src.separator.Path.mkdir")
def test_separator_demucs_success(mock_mkdir, mock_exists, mock_run):
    # Setup mock to simulate file existing before and after separation
    mock_exists.side_effect = [
        True, # input file exists
        True, # vocals exist
        True  # bgm exists
    ]
    
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_run.return_value = mock_result
    
    separator = AudioSeparator(mode="demucs")
    res = separator.separate(Path("data/audio/original/test.wav"))
    
    assert res is not None
    assert res.provider == "demucs"
    assert "vocals.wav" in str(res.vocals_path)
    assert "no_vocals.wav" in str(res.background_path)

@patch("src.separator.subprocess.run")
@patch("src.separator.Path.exists")
@patch("src.separator.Path.mkdir")
def test_separator_demucs_missing_outputs(mock_mkdir, mock_exists, mock_run):
    # Setup mock to simulate demucs succeeding but outputs are missing
    mock_exists.side_effect = [
        True, # input file exists
        False, # vocals missing
        False  # bgm missing
    ]
    
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_run.return_value = mock_result
    
    separator = AudioSeparator(mode="demucs")
    with pytest.raises(SeparationError, match="expected output files are missing"):
        separator.separate(Path("data/audio/original/test.wav"))
