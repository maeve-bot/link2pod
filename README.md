# Link2Pod

Convert any web page to an audio podcast using a 3-step pipeline:

1. **Fetch** - Download web page as HTML and extract main content
2. **LLM** - Process content into a podcast script using OpenRouter
3. **Voice** - Convert transcript to audio using TTS (Kokoro or Qwen3)

## Installation

```bash
# Install system dependency
# Ubuntu/Debian:
sudo apt install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Download Kokoro model files (for offline TTS)
# Place these in the link2pod directory:
# - kokoro-v1.0.onnx (~325MB)
# - voices-v1.0.bin (~28MB)
# Download from: https://github.com/thewh1teagle/kokoro-onnx/releases
```

## Usage

```bash
# Basic usage with Kokoro (default, works offline)
python link2pod.py https://example.com

# Specify output file (default is MP3)
python link2pod.py https://example.com -o my_podcast.wav

# Output as WAV instead of MP3
python link2pod.py https://example.com -o podcast.wav --wav

# Use a specific voice
python link2pod.py https://example.com -v am_michael

# List available Kokoro voices
python link2pod.py https://example.com --list-voices

# Use Qwen3 TTS (requires GPU)
python link2pod.py https://example.com --engine qwen3 --voice Serena

# Process a local file instead of URL
python link2pod.py ./article.md
python link2pod.py ./notes.txt -o podcast.wav

# Use Playwright for JavaScript-rendered pages
python link2pod.py https://example.com --playwright

# Limit content length (default: 10000 characters)
python link2pod.py https://example.com --max-length 5000

# Custom MP3 bitrate
python link2pod.py https://example.com --bitrate 128k
```

## TTS Engines

### Kokoro (default)
- **Pros**: Works offline, CPU-based, fast
- **Cons**: Limited voice variety
- **Model files needed**: `kokoro-v1.0.onnx`, `voices-v1.0.bin`

### Qwen3 (GPU)
- **Pros**: Best quality, more voice options, voice cloning
- **Cons**: Requires GPU with CUDA
- **Install**: `pip install qwen-tts`

## Architecture

```
link2pod/
├── link2pod.py           # Main entry point
├── src/
│   ├── fetch.py         # Web page fetching
│   ├── llm.py          # OpenRouter LLM processing
│   └── tts/
│       ├── __init__.py # Factory function
│       ├── base.py     # TTS interface
│       ├── kokoro.py   # Kokoro engine
│       ├── processor.py # Pause tag parsing
│       └── qwen.py     # Qwen3 engine
├── stt/                  # Optional: Speech-to-text (requires model)
└── requirements.txt
```

## API Key

The LLM step requires an OpenRouter API key. Set it via environment variable:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Custom Model Paths

If your model files are not in the link2pod directory, set environment variables:

```bash
# Kokoro models (default looks in link2pod/ directory)
export KOKORO_MODEL_PATH=/path/to/kokoro-v1.0.onnx
export KOKORO_VOICES_PATH=/path/to/voices-v1.0.bin
```

Or pass them programmatically:
```python
from src.tts import create_tts_engine

engine = create_tts_engine("kokoro", voice="af_sarah")
# Then modify engine._model_path and engine._voices_path before synthesis
```
