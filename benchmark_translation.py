import json
import time
from pathlib import Path
from src.translator import GroqTranslator
from src.utils.paths import TRANSCRIPTS_DIR

def main():
    transcript_path = TRANSCRIPTS_DIR / "Acids Bases and Salts in 30 Minutes ✅ ｜｜ Class 10 ｜｜ Fast Revision ｜｜ Alakh Pandey.json"
    
    if not transcript_path.exists():
        print(f"Transcript not found at {transcript_path}")
        return
        
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    all_segments = data.get("segments", [])
    if not all_segments:
        print("No segments found.")
        return
        
    # Take a subset of 50 segments for a quick benchmark
    subset = all_segments[:50]
    
    print(f"--- BENCHMARKING ADAPTIVE TRANSLATOR ---")
    print(f"Total segments selected: {len(subset)}")
    print(f"Total characters: {sum(len(s['text']) for s in subset)}")
    
    translator = GroqTranslator(max_retries=3)
    
    # We will track exactly what it does
    start_time = time.time()
    
    try:
        translated = translator._translate_segments(subset, "Hindi", "benchmark_30min")
        duration = time.time() - start_time
        print("\n--- RESULTS ---")
        print("Status: SUCCESS")
        print(f"Wall time: {duration:.2f} seconds")
        print(f"Output segments: {len(translated)}")
    except Exception as e:
        duration = time.time() - start_time
        print("\n--- RESULTS ---")
        print("Status: FAILED")
        print(f"Wall time: {duration:.2f} seconds")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
