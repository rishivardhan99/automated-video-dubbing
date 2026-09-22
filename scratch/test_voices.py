import asyncio
import edge_tts
from pathlib import Path

text = "Welcome to the summary of The Power of Your Subconscious Mind. In this video, we will explore how your thoughts shape your reality."

voices = [
    ("Christopher", "en-US-ChristopherNeural"), # Deep, audiobook style
    ("Steffan", "en-US-SteffanNeural"),         # Very clear and articulate
    ("Ryan", "en-GB-RyanNeural"),               # British, sophisticated
    ("Eric", "en-US-EricNeural"),               # Solid, engaging
    ("Roger", "en-US-RogerNeural"),             # Friendly, storytelling
    ("Brian", "en-US-BrianNeural"),             # Standard professional
]

async def main():
    out_dir = Path("scratch/voice_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating voice samples...")
    for name, voice_id in voices:
        print(f"Generating {name} ({voice_id})...")
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(str(out_dir / f"{name}.mp3"))
    print(f"Done! Samples saved to {out_dir.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
