"""
Kokoro TTS engine - offline, CPU-based neural TTS.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .base import TTSEngine
from .processor import (
    DEFAULT_PAUSE_MS,
    SplitLevel,
    SPLIT_LEVEL_PAUSES,
)


# Default paths (can be overridden via environment variables)
SCRIPT_DIR = Path(__file__).parent.parent.parent
DEFAULT_MODEL_PATH = Path(os.environ.get("KOKORO_MODEL_PATH", SCRIPT_DIR / "kokoro-v1.0.onnx"))
DEFAULT_VOICES_PATH = Path(os.environ.get("KOKORO_VOICES_PATH", SCRIPT_DIR / "voices-v1.0.bin"))


# Available Kokoro voices
KOKORO_VOICES = {
    # American voices
    "af_sarah": "American female, Sarah - warm and professional",
    "af_nicole": "American female, Nicole - friendly and casual",
    "af_jenny": "American female, Jenny - clear and articulate",
    "am_michael": "American male, Michael - deep and authoritative",
    "am_fen": "American male, Fen - young and energetic",
    "bm_george": "British male, George - sophisticated",
    "bm_lewis": "British male, Lewis - deep and measured",
    "bf_isabella": "British female, Isabella - elegant and refined",
    "bf_emma": "British female, Emma - natural and engaging",
    "bf_rebecca": "British female, Rebecca - professional",
    "af_aoife": "Irish female, Aoife - melodic",
    "am_adam": "American male, Adam - casual",
    "am_ryan": "American male, Ryan - upbeat",
    "bf_sophie": "British female, Sophie - modern",
    "bm_f不成": "Chinese male - Cantonese",
}


class KokoroEngine(TTSEngine):
    """Kokoro ONNX-based TTS engine."""
    
    def __init__(
        self,
        voice: str = "af_sarah",
        model_path: Optional[Path] = None,
        voices_path: Optional[Path] = None
    ):
        self._voice = voice
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._voices_path = voices_path or DEFAULT_VOICES_PATH
        self._kokoro = None
        
        # Validate voice
        if voice not in self.list_voices():
            raise ValueError(f"Unknown voice: {voice}. Available: {self.list_voices()}")
    
    @property
    def name(self) -> str:
        return "kokoro"
    
    @property
    def default_voice(self) -> str:
        return "af_sarah"
    
    def list_voices(self) -> list[str]:
        return list(KOKORO_VOICES.keys())
    
    def _load_model(self):
        """Lazy load the Kokoro model."""
        if self._kokoro is not None:
            return
        
        from kokoro_onnx import Kokoro
        
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Kokoro model not found at {self._model_path}. "
                "Please download from https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        
        if not self._voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro voices not found at {self._voices_path}. "
                "Please download from https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        
        self._kokoro = Kokoro(
            model_path=str(self._model_path),
            voices_path=str(self._voices_path)
        )
    
    def _synthesize_single(self, text: str) -> tuple:
        """Synthesize a single text chunk."""
        self._load_model()
        samples, sample_rate = self._kokoro.create(text, voice=self._voice)
        return samples, sample_rate
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize text to speech audio file.
        
        Args:
            text: Text to synthesize (pause tags are optional, now deprecated)
            output_path: Output wav file path
        
        Returns:
            Path to the generated audio file
        """
        # Chunk text with split level tracking
        chunks = self._chunk_text_with_levels(text, max_size=1000)
        
        # Synthesize each chunk with pauses between
        audio_arrays = []
        sample_rate = None
        
        for i, (chunk_text, split_level) in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            
            # Synthesize this chunk
            audio, sr = self._synthesize_single(chunk_text)
            audio_arrays.append(audio)
            sample_rate = sr
            
            # Add pause based on split level (not between last chunk)
            if i < len(chunks) - 1 and sample_rate is not None:
                pause_ms = SPLIT_LEVEL_PAUSES.get(split_level, DEFAULT_PAUSE_MS)
                pause_samples = int(sample_rate * pause_ms / 1000)
                pause = np.zeros(pause_samples, dtype=np.float32)
                audio_arrays.append(pause)
        
        if not audio_arrays:
            raise ValueError("No audio generated")
        
        # Concatenate all audio with pauses
        final_audio = np.concatenate(audio_arrays)
        
        # Save to file
        if output_path is None:
            output_path = "output.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio_int16 = (final_audio * 32767).astype(np.int16)
        sf.write(output_path, audio_int16, sample_rate)
        
        return str(output_path)
    
    def _chunk_text_with_levels(self, text: str, max_size: int = 1000) -> list[tuple[str, str]]:
        """
        Split text into chunks with split level tracking for pause determination.
        
        Returns list of (text, split_level) tuples where split_level is:
        - PARAGRAPH: split at double newline (1000ms pause)
        - SENTENCE: split at .!? (800ms pause)  
        - WORD: force split at max_size (500ms pause)
        
        Larger chunks = better intonation, so we prefer paragraphs > sentences > words.
        """
        import re
        
        chunks_with_levels = []
        
        # Step 1: Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            # If paragraph fits within max_size, keep it whole
            if para_len <= max_size:
                chunks_with_levels.append((para, SplitLevel.PARAGRAPH))
                continue
            
            # Step 2: Paragraph too long - split by sentences
            sentence_endings = re.compile(r"(?<=[.!?])\s+")
            sentences = sentence_endings.split(para)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_len = len(sentence)
                
                # If adding this sentence would exceed limit, start new chunk
                if current_length + sentence_len > max_size and current_chunk:
                    chunks_with_levels.append((" ".join(current_chunk), SplitLevel.SENTENCE))
                    current_chunk = []
                    current_length = 0
                
                # If single sentence exceeds max_size, split by words
                if sentence_len > max_size:
                    # First, flush current chunk if any
                    if current_chunk:
                        chunks_with_levels.append((" ".join(current_chunk), SplitLevel.SENTENCE))
                        current_chunk = []
                        current_length = 0
                    
                    # Split long sentence by words
                    words = sentence.split()
                    word_chunk = []
                    word_len = 0
                    
                    for word in words:
                        word_len_inc = len(word) + 1  # +1 for space
                        if word_len + word_len_inc > max_size and word_chunk:
                            chunks_with_levels.append((" ".join(word_chunk), SplitLevel.WORD))
                            word_chunk = []
                            word_len = 0
                        word_chunk.append(word)
                        word_len += word_len_inc
                    
                    if word_chunk:
                        current_chunk = word_chunk
                        current_length = word_len
                    continue
                
                # Normal sentence - add to current chunk
                current_chunk.append(sentence)
                current_length += sentence_len + 1
            
            # Flush remaining chunk
            if current_chunk:
                chunks_with_levels.append((" ".join(current_chunk), SplitLevel.SENTENCE))
        
        return chunks_with_levels if chunks_with_levels else [(text, SplitLevel.SENTENCE)]
    
    def synthesize_chunks(self, chunks: list[str], output_path: Optional[str] = None) -> str:
        """Synthesize multiple chunks and concatenate."""
        audio_arrays = []
        sample_rate = None
        
        for chunk in chunks:
            audio, sr = self._synthesize_single(chunk)
            audio_arrays.append(audio)
            sample_rate = sr
        
        # Concatenate with pauses
        pause_samples = int(sample_rate * 0.5)  # 500ms pause
        pause = np.zeros(pause_samples, dtype=np.float32)
        
        final_audio = []
        for i, audio in enumerate(audio_arrays):
            if i > 0:
                final_audio.append(pause)
            final_audio.append(audio)
        
        final_audio = np.concatenate(final_audio)
        
        # Save to file
        if output_path is None:
            output_path = "output.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio_int16 = (final_audio * 32767).astype(np.int16)
        sf.write(output_path, audio_int16, sample_rate)
        
        return str(output_path)
