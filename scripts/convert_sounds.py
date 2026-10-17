"""Convert alert sounds from MP3 to WAV using pygame.

Reads each .mp3 from sounds/, decodes it via pygame.mixer, and writes a
16-bit mono 44100 Hz .wav alongside it.  The AudioPlayer loader tries
.ogg → .wav → .mp3, so having .wav files removes the MP3-deprecation warning.

Usage (run once after cloning, requires audio hardware or a virtual device):
    python scripts/convert_sounds.py

Note: if ffmpeg is available, prefer:
    for f in sounds/*.mp3; do ffmpeg -i "$f" "${f%.mp3}.ogg"; done
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

SOUNDS_DIR = Path(__file__).parent.parent / "sounds"
NAMES = ["drowsy", "left", "right", "down"]


def convert() -> None:
    import pygame  # noqa: PLC0415

    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    try:
        for name in NAMES:
            src = SOUNDS_DIR / f"{name}.mp3"
            dst = SOUNDS_DIR / f"{name}.wav"
            if not src.exists():
                print(f"  skip {src} (not found)")
                continue
            snd = pygame.mixer.Sound(str(src))
            raw = pygame.sndarray.array(snd)
            with wave.open(str(dst), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(raw.tobytes())
            print(f"  {src.name} -> {dst.name} ({os.path.getsize(dst):,} bytes)")
    finally:
        pygame.mixer.quit()


if __name__ == "__main__":
    print("Converting sounds/ MP3 -> WAV ...")
    convert()
    print("Done. Re-run main.py -- AudioPlayer will load .wav files (no MP3 warning).")
