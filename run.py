import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.audio import (
    AudioExtractionError,
    extract_audio,
    get_audio_duration,
    extract_benchmark_audio,
    mux_audio_video,
)

from src.downloader import (
    DownloadError,
    download_video,
)

from src.transcribers import (
    GroqTranscriber,
    TranscriptionError,
)

from src.separator import (
    AudioSeparator,
    SeparationError,
)

from src.translator import (
    GroqTranslator,
    TranslationError,
)

from src.synthesizer import (
    EdgeTTSSynthesizer,
    SynthesisError,
)

from src.quality import (
    analyze_transcript,
    QualityState,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Automated Video Dubbing Pipeline",
    )

    parser.add_argument(
        "input",
        help=(
            "YouTube URL or path to a local "
            "audio file (for benchmark mode)"
        ),
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run transcription benchmark on the "
            "first 60 seconds of the input audio"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Proceed to translation even if "
            "transcript quality checks fail"
        ),
    )

    args = parser.parse_args()

    timings = {}

    try:
        print()
        print("=" * 60)
        print("AUTOMATED VIDEO DUBBING")
        print("=" * 60)
        print()

        total_start = time.time()

        if args.benchmark:
            _run_benchmark(args, timings)
        else:
            _run_pipeline(args, timings, total_start)

    except (
        DownloadError,
        AudioExtractionError,
        SeparationError,
        TranscriptionError,
        TranslationError,
        SynthesisError,
    ) as exc:

        print()
        print(f"ERROR: {exc}")
        sys.exit(1)


def _run_benchmark(args, timings):
    """
    Benchmark mode: extract first 60 seconds,
    transcribe with Groq STT, run quality analysis.
    """

    audio_path = Path(args.input)

    if not audio_path.exists():
        print(
            f"ERROR: File does not exist: "
            f"{audio_path}"
        )
        sys.exit(1)

    print("--- BENCHMARK MODE ---")
    print()

    # Step 1: Extract 60s clip.
    t0 = time.time()
    benchmark_audio = extract_benchmark_audio(
        audio_path,
        max_duration_s=60,
    )
    audio_duration = get_audio_duration(
        benchmark_audio,
    )
    timings["Audio Extraction"] = time.time() - t0

    # Step 2: Transcribe with Groq STT.
    t0 = time.time()
    transcriber = GroqTranscriber()
    transcript_path = transcriber.transcribe(
        benchmark_audio,
    )
    transcription_time = time.time() - t0
    timings["Groq STT"] = transcription_time

    # Step 3: Load and analyze quality.
    with transcript_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        transcript_data = json.load(f)

    report = analyze_transcript(
        transcript_data,
        audio_duration,
        transcription_time,
    )
    report.log(logger)
    
    # Step 4: Translate
    t0 = time.time()
    translator = GroqTranslator(
        max_input_chars=5000,
        max_retries=5,
    )
    translation_path = translator.translate(
        transcript_path,
    )
    timings["Translation"] = time.time() - t0

    # Step 5: Synthesize (Edge-TTS + Assembly)
    t0 = time.time()
    synthesizer = EdgeTTSSynthesizer()
    output_audio, meta_report = synthesizer.synthesize(
        translation_path,
        benchmark_audio,
    )
    timings["Synthesis & Assembly"] = time.time() - t0

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Provider:        groq"
    )
    print(
        f"Model:           {transcriber.model_name}"
    )
    print(
        f"Language:        "
        f"{transcriber.language or 'auto-detect'}"
    )
    print(
        f"Original Audio:  {audio_path}"
    )
    print(
        f"Benchmark Audio: {benchmark_audio}"
    )
    print(
        f"Transcript:      {transcript_path}"
    )
    print(
        f"Translation:     {translation_path}"
    )
    print(
        f"Dubbed Audio:    {output_audio}"
    )
    
    print("\n--- Synthesis Benchmarks ---")
    
    total_raw_tts = sum(m.raw_tts_duration for m in meta_report)
    total_trimmed = sum(m.trimmed_tts_duration for m in meta_report)
    avg_speed = sum(m.applied_speed_factor for m in meta_report) / max(len(meta_report), 1)
    violations = [m for m in meta_report if m.timing_violation]
    truncated = [m for m in meta_report if m.truncated]
    largest_overflow = max((m.overflow_ms for m in meta_report), default=0)
    
    print(f"Source Duration:          {audio_duration:.2f}s")
    print(f"Number of Segments:       {len(meta_report)}")
    print(f"Total Raw TTS Duration:   {total_raw_tts:.2f}s")
    print(f"Total Trimmed Duration:   {total_trimmed:.2f}s")
    print(f"Average Speed Factor:     {avg_speed:.2f}x")
    print(f"Timing Violations:        {len(violations)}")
    print(f"Truncated Segments:       {len(truncated)}")
    print(f"Largest Overflow:         {largest_overflow}ms")

    if violations:
        print("\n--- Timing Violations Details ---")
        for v in violations:
            msg = f"  [!] Segment {v.id}: Target {v.target_duration:.2f}s, Trimmed TTS {v.trimmed_tts_duration:.2f}s, Available {v.available_duration:.2f}s."
            if v.truncated:
                msg += f" TRUNCATED by {v.overflow_ms}ms."
            else:
                msg += f" Capped at {v.applied_speed_factor:.2f}x."
            print(msg)

    print("\n--- Performance ---")
    for stage, t in timings.items():
        print(f"{stage:18}: {t:6.1f}s")

    if audio_duration > 0:
        print(
            f"{'Audio Duration':18}: "
            f"{audio_duration:6.1f}s"
        )


def _run_pipeline(args, timings, total_start):
    """
    Full pipeline: download → audio → Groq STT →
    quality gate → Groq translation.
    """

    url = args.input

    # --------------------------------------------------
    # STEP 1 — DOWNLOAD
    # --------------------------------------------------

    t0 = time.time()
    from src.downloader import validate_youtube_url
    if validate_youtube_url(url):
        video_path = download_video(url)
    else:
        video_path = Path(url)
        if not video_path.exists():
            raise DownloadError(f"Local file does not exist: {video_path}")
        logger.info("Using local video file: %s", video_path)
        
    timings["Download"] = time.time() - t0

    print()

    # --------------------------------------------------
    # STEP 2 — AUDIO EXTRACTION
    # --------------------------------------------------

    t0 = time.time()
    audio_path = extract_audio(video_path)
    audio_duration = get_audio_duration(audio_path)
    timings["Audio Extraction"] = time.time() - t0

    print()

    # --------------------------------------------------
    # STEP 2.5 — VOCAL SEPARATION
    # --------------------------------------------------
    
    t0 = time.time()
    separator = AudioSeparator(mode=os.getenv("AUDIO_SEPARATION", "none"))
    sep_result = separator.separate(audio_path)
    
    if sep_result:
        timings["Vocal Separation"] = sep_result.processing_time
        
        # Determine transcription input based on config
        stt_mode = os.getenv("TRANSCRIPTION_AUDIO", "original").lower()
        if stt_mode == "separated_vocals":
            logger.info("Using separated vocals for transcription.")
            transcription_input = sep_result.vocals_path
        else:
            logger.info("Using original audio for transcription.")
            transcription_input = audio_path
            
        bg_music_path = sep_result.background_path
    else:
        transcription_input = audio_path
        bg_music_path = None
        
    print()

    # --------------------------------------------------
    # STEP 3 — GROQ TRANSCRIPTION
    # --------------------------------------------------

    t0 = time.time()
    transcriber = GroqTranscriber()
    transcript_path = transcriber.transcribe(
        transcription_input,
    )
    transcription_time = time.time() - t0
    timings["Groq STT"] = transcription_time

    print()

    # --------------------------------------------------
    # STEP 3.5 — QUALITY GATE
    # --------------------------------------------------

    with transcript_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        transcript_data = json.load(f)

    report = analyze_transcript(
        transcript_data,
        audio_duration,
        transcription_time,
    )
    report.log(logger)

    if (
        report.state == QualityState.FAIL
        and not args.force
    ):
        print()
        print(
            "[!] Pipeline stopped: Transcript "
            "failed quality checks."
        )
        print(
            "Use --force to bypass this gate "
            "for testing."
        )
        sys.exit(1)

    # --------------------------------------------------
    # STEP 4 — GROQ TRANSLATION
    # --------------------------------------------------

    t0 = time.time()
    translator = GroqTranslator(
        max_retries=5,
    )
    translation_path = translator.translate(
        transcript_path,
    )
    timings["Translation"] = time.time() - t0

    print()
    
    # --------------------------------------------------
    # STEP 5 — TTS & ASSEMBLY
    # --------------------------------------------------

    t0 = time.time()
    synthesizer = EdgeTTSSynthesizer()
    output_audio, meta_report = synthesizer.synthesize(
        translation_path,
        audio_path,
        bg_music_path=bg_music_path,
    )
    timings["Synthesis & Assembly"] = time.time() - t0
    
    print()

    # --------------------------------------------------
    # STEP 6 — VIDEO MUXING
    # --------------------------------------------------

    t0 = time.time()
    final_video = mux_audio_video(
        video_path,
        output_audio,
    )
    timings["Video Muxing"] = time.time() - t0

    print()

    total_time = time.time() - total_start

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("=" * 60)
    print("PIPELINE COMPLETE SUCCESSFUL")
    print("=" * 60)
    print()
    print(f"Original Video: {video_path}")
    print(f"Final Video:    {final_video}")
    print(f"Original Audio: {audio_path}")
    print(f"Transcript:     {transcript_path}")
    print(f"Translation:    {translation_path}")
    print(f"Dubbed Audio:   {output_audio}")

    print("\n--- Performance Report ---")

    for stage, t in timings.items():
        print(f"{stage:18}: {t:6.1f}s")

    print(f"{'Total':18}: {total_time:6.1f}s")

    if audio_duration > 0:
        print(
            f"{'Audio Duration':18}: "
            f"{audio_duration:6.1f}s"
        )
        ratio = total_time / audio_duration
        print(
            f"{'Processing Ratio':18}: "
            f"{ratio:6.2f}x"
        )


if __name__ == "__main__":
    main()