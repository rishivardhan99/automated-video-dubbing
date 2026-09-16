# Automated Video Dubbing

A modular pipeline for automatically dubbing YouTube videos into English.

## Architecture

```
YouTube URL
→ yt-dlp (download)
→ FFmpeg (audio extraction)
→ Groq Whisper STT (transcription)
→ Transcript Quality Gate (PASS/WARN/FAIL)
→ Groq GPT-OSS-20B (translation)
→ Edge-TTS (speech synthesis)
→ FFmpeg (final mux)
```

## Setup

### Prerequisites

- Python 3.10+
- `ffmpeg` installed and available on PATH

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_api_key_here

# Optional overrides (shown with defaults)
GROQ_MODEL=openai/gpt-oss-20b
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_STT_LANGUAGE=
```

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API authentication | *required* |
| `GROQ_MODEL` | Translation LLM model | `openai/gpt-oss-20b` |
| `GROQ_STT_MODEL` | Speech-to-text model | `whisper-large-v3-turbo` |
| `GROQ_STT_LANGUAGE` | Source language code (e.g. `te`, `hi`) | auto-detect |

## Usage

### 1. Benchmark Mode

Test Groq STT quality on the first 60 seconds of an audio file before committing to a full run:

```bash
python run.py --benchmark "data/audio/your_audio_file.wav"
```

This will:
- Extract a 60-second clip via FFmpeg
- Transcribe with Groq Whisper STT
- Run the quality gate analysis
- Print diagnostics (language, segments, timing)
- Save a benchmark transcript JSON

### 2. Full Pipeline

The script accepts **both** YouTube URLs and local video files. If you pass a YouTube URL, it will download it. If you pass a local file, it will skip the download step and use your file directly.

```bash
# Dub a YouTube video
python run.py "https://www.youtube.com/watch?v=..."

# Dub a local video file
python run.py "data/input/my_video.mp4"
```

If the transcript fails quality checks, the pipeline halts before making translation API calls. To force it to proceed (for testing):

```bash
python run.py --force "https://www.youtube.com/watch?v=..."
```

### 3. Long Videos (30min / 2hr)

The pipeline automatically handles large audio files that exceed the Groq API's 25 MB upload limit:

- Audio is split into sequential 10-minute chunks via FFmpeg
- Each chunk is transcribed independently
- Timestamps are offset to maintain chronological correctness
- Segments are merged and renumbered sequentially
- Temporary chunks are cleaned up after processing

No manual intervention is required.

## Quality Gate

Before translation, every transcript passes through a quality analysis that evaluates:

| Check | State |
|---|---|
| Empty transcript | FAIL |
| Pathological repetition (>40%) | FAIL |
| Excessive single-char segments (>30%) | FAIL |
| Invalid timestamps (end < start) | FAIL |
| Excessive segment density (>1.5/s) | FAIL |
| Intra-segment hallucination loops | FAIL |
| Script mixing + structural corruption | FAIL |
| High repetition (>20%) | WARN |
| Multiple script families detected | WARN |
| Low language confidence (<0.5) | WARN |
| Near-zero duration segments | WARN |

## Testing

Run the offline unit tests (no API keys required):

```bash
python -m pytest tests/ -v
```

## Rate Limits

Groq applies rate limits to both STT and chat completions. For long videos with many segments:

- The translator uses exponential backoff with retry-after header parsing
- Large audio files are chunked to stay within upload limits
- Monitor Groq dashboard for quota usage

## Test Videos

The system has been designed and tested to handle long-form content, including the following benchmark videos:
- **30 Minute Benchmark**: [https://youtu.be/rgjb5Ubh90k?si=9oGMokSFtAJ6vuST](https://youtu.be/rgjb5Ubh90k?si=9oGMokSFtAJ6vuST)
- **2 Hour Benchmark**: [https://youtu.be/RGKi6LSPDLU?si=Jps-EUb4Ej4JVUjY](https://youtu.be/RGKi6LSPDLU?si=Jps-EUb4Ej4JVUjY)
