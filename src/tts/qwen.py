"""
Qwen3 TTS engine - GPU-based neural TTS with voice customization.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import tqdm

from .base import TTSEngine


# Available Qwen3 voices (from CustomVoice model)
QWEN3_VOICES = [
    "Ryan",      # American male - deep, authoritative
    "Serena",   # American female - warm, engaging
    "Vivian",   # British female - elegant, refined
    "Uncle_Fu", # Chinese male - wise, elder
    "Dylan",    # American male - young, casual
    "Eric",     # American male - professional
    "Anna",     # American female - clear
    "Lisa",     # American female - friendly
    "Emma",     # British female - natural
]


class Qwen3Engine(TTSEngine):
    """Qwen3 TTS engine - requires GPU."""
    
    def __init__(
        self,
        voice: str = "Ryan",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda",
        language: str = "English"
    ):
        self._voice = voice
        self._model_name = model_name
        self._device = device
        self._language = language
        self._model = None
        
        # Validate voice
        if voice not in self.list_voices():
            raise ValueError(f"Unknown voice: {voice}. Available: {self.list_voices()}")
    
    @property
    def name(self) -> str:
        return "qwen3"
    
    @property
    def default_voice(self) -> str:
        return "Ryan"
    
    def list_voices(self) -> list[str]:
        return QWEN3_VOICES.copy()
    
    def _load_model(self):
        """Lazy load the Qwen3 model."""
        if self._model is not None:
            return
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen-tts not installed. Install with: pip install qwen-tts"
            )
        
        print(f"Loading {self._model_name} on {self._device}...")
        
        dtype = torch.bfloat16 if "cuda" in self._device else torch.float32
        attn_impl = "flash_attention_2" if "cuda" in self._device else "sdpa"
        
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_name,
            device_map=self._device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print("Model loaded!")
    
    def _synthesize_single(self, text: str) -> tuple:
        """Synthesize a single text chunk."""
        self._load_model()
        
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._voice,
            non_streaming_mode=True,
        )
        
        return wavs[0], sr
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize text to speech audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output wav file path
        
        Returns:
            Path to the generated audio file
        """
        # Chunk the text
        chunks = self._chunk_text(text)
        
        if len(chunks) == 1:
            # Single chunk
            audio, sr = self._synthesize_single(text)
        else:
            # Multiple chunks - use batch synthesis
            audio, sr = self.synthesize_chunks(chunks)
        
        # Save to file
        if output_path is None:
            output_path = "output.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio_int16 = (audio * 32767).astype(np.int16)
        sf.write(output_path, audio_int16, sr)
        
        return str(output_path)
    
    def synthesize_chunks(self, chunks: list[str], output_path: Optional[str] = None) -> tuple:
        """
        Synthesize multiple chunks using batch processing for efficiency.
        
        Returns:
            (audio_array, sample_rate)
        """
        self._load_model()
        
        audio_arrays = []
        sample_rate = None
        
        # Batch synthesis (16 at a time)
        batch_size = 16
        for i in tqdm.tqdm(range(0, len(chunks), batch_size), desc="Synthesizing", unit="batch"):
            batch_texts = chunks[i:i + batch_size]
            
            wavs, sr = self._model.generate_custom_voice(
                text=batch_texts,
                language=self._language,
                speaker=self._voice,
                non_streaming_mode=True,
            )
            
            # Handle different return types
            if isinstance(wavs, np.ndarray):
                if len(wavs.shape) == 1:
                    wavs = [wavs]
                elif len(wavs.shape) > 1:
                    wavs = list(wavs)
            
            audio_arrays.extend(wavs)
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
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            audio_int16 = (final_audio * 32767).astype(np.int16)
            sf.write(output_path, audio_int16, sample_rate)
        
        return final_audio, sample_rate
    
    def _chunk_text(self, text: str, max_size: int = 1000) -> list[str]:
        """
        Split text into chunks with priority:
        1. Paragraphs (double newlines) - natural pause points
        2. Sentences (if paragraph > max_size)
        3. Words (if sentence > max_size)
        
        This gives larger, more cohesive chunks (2-4 sentences) which helps
        the TTS maintain better intonation and flow.
        """
        import re
        
        # Step 1: Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            # If paragraph fits within max_size, keep it whole
            if para_len <= max_size:
                chunks.append(para)
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
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # If single sentence exceeds max_size, split by words
                if sentence_len > max_size:
                    # First, flush current chunk if any
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Split long sentence by words
                    words = sentence.split()
                    word_chunk = []
                    word_len = 0
                    
                    for word in words:
                        word_len_inc = len(word) + 1  # +1 for space
                        if word_len + word_len_inc > max_size and word_chunk:
                            chunks.append(" ".join(word_chunk))
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
                chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]
