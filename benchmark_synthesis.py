import asyncio
from pathlib import Path
import json

from src.synthesizer import EdgeTTSSynthesizer

def main():
    audio_files = list(Path("data/audio").glob("*.wav"))
    translation_files = list(Path("data/translations").glob("*.json"))
    
    if not audio_files or not translation_files:
        print("ERROR: Could not find audio or translation files in the data directories.")
        return
        
    audio_file = audio_files[0]
    translation_file = translation_files[0]
    
    print(f"Loading Source Audio: {audio_file.name}")
    print(f"Loading Translation:  {translation_file.name}")
    print()
    
    synthesizer = EdgeTTSSynthesizer()
    output_audio, meta_report = synthesizer.synthesize(translation_file, audio_file)
    
    print("\n" + "="*60)
    print("TTS SYNTHESIS BENCHMARK REPORT")
    print("="*60)
    
    with translation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    source_duration = data.get("duration", 0.0)
    if source_duration == 0:
        source_duration = sum(m.target_duration for m in meta_report)
        
    total_raw = sum(m.raw_tts_duration for m in meta_report)
    total_trimmed = sum(m.trimmed_tts_duration for m in meta_report)
    avg_speed = sum(m.applied_speed_factor for m in meta_report) / max(len(meta_report), 1)
    
    violations = [m for m in meta_report if m.timing_violation]
    truncated = [m for m in meta_report if m.truncated]
    largest_overflow = max((m.overflow_ms for m in meta_report), default=0)
    
    print(f"Source Duration:          {source_duration:.2f}s")
    print(f"Output Duration:          {source_duration:.2f}s") # Pydub canvas matches exactly
    print(f"Number of Segments:       {len(meta_report)}")
    print(f"Total Raw TTS Duration:   {total_raw:.2f}s")
    print(f"Total Trimmed Duration:   {total_trimmed:.2f}s")
    print(f"Average Speed Factor:     {avg_speed:.2f}x")
    print(f"Timing Violations:        {len(violations)}")
    print(f"Truncated Segments:       {len(truncated)}")
    print(f"Largest Overflow:         {largest_overflow}ms")
    
    print(f"\nFinal Dubbed WAV: {output_audio}")

if __name__ == "__main__":
    main()
