"""
TTS (Text-to-Speech) module.

Provides interchangeable TTS engines:
- Kokoro: Offline, CPU-based, good quality
- Qwen3: GPU-based, best quality, requires CUDA
"""

from abc import ABC, abstractmethod
from typing import Optional


class TTSEngine(ABC):
    """Base class for TTS engines."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the engine name."""
        pass
    
    @property
    @abstractmethod
    def default_voice(self) -> str:
        """Return the default voice name."""
        pass
    
    @abstractmethod
    def list_voices(self) -> list[str]:
        """List available voices for this engine."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize text to speech.
        
        Args:
            text: The text to synthesize
            output_path: Optional output file path (should be .wav)
        
        Returns:
            Path to the generated audio file
        """
        pass
    
    def synthesize_chunks(self, chunks: list[str], output_path: Optional[str] = None) -> str:
        """
        Synthesize multiple text chunks and concatenate them.
        
        Default implementation calls synthesize() for each chunk.
        Engines can override for batch processing.
        """
        import numpy as np
        import soundfile as sf
        
        audio_arrays = []
        sample_rate = None
        
        for i, chunk in enumerate(chunks):
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
    
    @abstractmethod
    def _synthesize_single(self, text: str) -> tuple:
        """Internal method to synthesize single text. Returns (audio_array, sample_rate)."""
        pass


from pathlib import Path
