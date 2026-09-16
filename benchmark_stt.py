import time
from pathlib import Path
from src.transcribers.groq_transcriber import GroqTranscriber
from src.utils.paths import AUDIO_DIR

def main():
    audio_path = AUDIO_DIR / "original" / "Acids Bases and Salts in 30 Minutes ✅ ｜｜ Class 10 ｜｜ Fast Revision ｜｜ Alakh Pandey.wav"
    
    if not audio_path.exists():
        print(f"Audio not found at {audio_path}. Please extract it first.")
        return
        
    print(f"--- BENCHMARKING MULTITHREADED STT (Concurrency 2) ---")
    
    # Force max concurrency to 2 for benchmark
    import os
    os.environ["STT_MAX_CONCURRENCY"] = "2"
    os.environ["GROQ_STT_LANGUAGE"] = ""
    
    transcriber = GroqTranscriber()
    
    start_time = time.time()
    
    try:
        # This will use the new chunking and ThreadPoolExecutor
        result = transcriber._transcribe_chunked(audio_path)
        duration = time.time() - start_time
        
        print("\n--- RESULTS ---")
        print("Status: SUCCESS")
        print(f"Wall time: {duration:.2f} seconds")
        print(f"Total chunks processed: {result.get('completed_chunks')}")
        print(f"Merged duration: {result.get('duration')}s")
        print(f"Total segments: {len(result.get('segments', []))}")
        
    except Exception as e:
        duration = time.time() - start_time
        print("\n--- RESULTS ---")
        print("Status: FAILED")
        print(f"Wall time: {duration:.2f} seconds")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
