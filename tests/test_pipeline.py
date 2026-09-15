from src.downloader import validate_youtube_url


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