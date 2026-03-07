#!/usr/bin/env python3
"""
STT validation script - transcribe audio to verify script quality.
"""

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str, model_size: str = "small") -> str:
    """Transcribe audio file using Faster Whisper."""
    print(f"Loading Whisper model ({model_size})...")
    
    # Run on CPU with int8 for efficiency
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print(f"Transcribing {audio_path}...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    full_text = []
    for segment in segments:
        full_text.append(segment.text.strip())
    
    return " ".join(full_text)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-m", "--model", default="small", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    try:
        transcript = transcribe_audio(args.audio, args.model)
        print("\n=== TRANSCRIPT ===")
        print(transcript)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
