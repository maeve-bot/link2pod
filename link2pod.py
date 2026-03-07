#!/usr/bin/env python3
"""
Link2Pod - Convert web pages or local files to audio podcasts

A 3-step pipeline:
1. Fetch: Download web page as HTML, or read local file
2. LLM: Process content into a podcast script using OpenRouter
3. Voice: Convert transcript to audio using TTS

Usage:
    python3 link2pod.py <url-or-file> [options]

Arguments:
    url-or-file           URL of a web page OR path to a local file (.txt, .md)

Options:
    -o, --output PATH     Output audio path (default: ./output/<domain>_<timestamp>.mp3)
    -e, --engine ENGINE  TTS engine: kokoro (default), qwen3
    -v, --voice VOICE    Voice name (see --list-voices)
    --list-voices        List available voices for the engine and exit
    --max-length         Max characters to process (default: 10000)
    --playwright         Use Playwright for JavaScript-rendered pages
    --wav                Output as WAV instead of MP3
    --bitrate            MP3 bitrate (default: 192k)

Examples:
    # From web page
    python3 link2pod.py https://example.com/article
    python3 link2pod.py https://news.ycombinator.com -o podcast.wav
    
    # From local file
    python3 link2pod.py /path/to/article.md
    python3 link2pod.py ./notes.txt -o my-podcast.mp3
    
    # TTS options
    python3 link2pod.py https://blog.example.com --engine qwen3 --voice Serena
    python3 link2pod.py https://example.com --list-voices
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.fetch import fetch_webpage, fetch_local_file, is_url
from src.llm import generate_podcast_script
from src.tts import create_tts_engine


def convert_to_mp3(wav_path: str, bitrate: str = "192k", mp3_path: str | None = None) -> str:
    """Convert WAV to MP3 using ffmpeg."""
    wav_path = Path(wav_path)
    if mp3_path is None:
        mp3_path = wav_path.with_suffix(".mp3")
    else:
        mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-b:a", bitrate,
        str(mp3_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return str(mp3_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert web pages or local files to audio podcasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("input", help="URL of the web page OR path to a local file (.txt, .md)")
    parser.add_argument("-o", "--output", help="Output audio path (default: MP3, unless --wav)")
    parser.add_argument("-e", "--engine", default="kokoro", 
                        choices=["kokoro", "qwen3"],
                        help="TTS engine to use (default: kokoro)")
    parser.add_argument("-v", "--voice", 
                        help="Voice name (see --list-voices)")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices for the engine and exit")
    parser.add_argument("--max-length", type=int, default=10000,
                        help="Max characters to process from page (default: 10000)")
    parser.add_argument("--playwright", action="store_true",
                        help="Use Playwright for JavaScript-rendered pages (slower but handles JS-heavy sites)")
    parser.add_argument("--wav", action="store_true",
                        help="Output as WAV instead of MP3")
    parser.add_argument("--bitrate", default="192k",
                        help="MP3 bitrate (default: 192k)")
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        engine = create_tts_engine(args.engine)
        voices = engine.list_voices()
        print(f"Available voices for {args.engine}:")
        for v in voices:
            print(f"  {v}")
        return
    
    input_arg = args.input
    output_path = args.output
    engine_name = args.engine
    voice = args.voice
    max_length = args.max_length
    use_playwright = args.playwright
    
    # Detect input type
    if is_url(input_arg):
        input_type = "URL"
        input_display = input_arg
    else:
        input_type = "file"
        input_display = str(Path(input_arg).resolve())
    
    print(f"=== Link2Pod ===")
    print(f"Input: {input_type} - {input_display}")
    print(f"Engine: {engine_name}")
    if input_type == "URL":
        print(f"Playwright: {use_playwright}")
    print()
    
    # Step 1: Fetch content (web page or local file)
    print("=== STEP 1: FETCH ===")
    try:
        if is_url(input_arg):
            content = fetch_webpage(input_arg, max_length=max_length, use_playwright=use_playwright)
            print(f"Fetched {len(content)} characters from web page")
        else:
            content = fetch_local_file(input_arg, max_length=max_length)
            print(f"Read {len(content)} characters from file")
    except Exception as e:
        print(f"Error fetching content: {e}")
        sys.exit(1)
    
    # Step 2: Generate podcast script
    print("\n=== STEP 2: LLM PROCESSING ===")
    try:
        # Use input as the reference for the transcript
        transcript = generate_podcast_script(content, input_display)
        print(f"Generated transcript: {len(transcript)} characters")
    except Exception as e:
        print(f"Error generating script: {e}")
        sys.exit(1)
    
    # Save transcript for reference
    # Default output is MP3 (or WAV with --wav flag)
    if not output_path:
        from datetime import datetime
        # Create a safe filename from the input
        if input_type == "URL":
            from urllib.parse import urlparse
            domain = urlparse(input_arg).netloc.replace('.', '_')
        else:
            # Use filename without extension
            domain = Path(input_arg).stem.replace('.', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = ".wav" if args.wav else ".mp3"
        output_path = f"output/{domain}_{timestamp}{extension}"
    
    output_path = Path(output_path)
    transcript_path = output_path.with_suffix('.txt')
    
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcript)
    print(f"Transcript saved to: {transcript_path}")
    
    # Step 3: Synthesize audio
    print("\n=== STEP 3: VOICE SYNTHESIS ===")
    try:
        engine = create_tts_engine(engine_name, voice=voice)
        temp_wav_path = output_path.with_suffix(".tmp.wav")
        audio_path = Path(engine.synthesize(transcript, str(temp_wav_path)))
        print(f"Intermediate WAV saved to: {audio_path}")
    except Exception as e:
        print(f"Error synthesizing audio: {e}")
        sys.exit(1)
    
    # Step 4: Convert to MP3 unless --wav is specified
    if args.wav:
        audio_path.replace(output_path)
        print(f"\nWAV saved to: {output_path}")
    else:
        print("\n=== STEP 4: CONVERT TO MP3 ===")
        try:
            audio_path = Path(convert_to_mp3(str(audio_path), bitrate=args.bitrate, mp3_path=str(output_path)))
            if temp_wav_path.exists():
                temp_wav_path.unlink()
            print(f"MP3 saved to: {audio_path}")
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            sys.exit(1)
    
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
