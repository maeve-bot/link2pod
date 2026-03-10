"""
TTS module - factory function to create TTS engines.
"""

from typing import Optional

from .base import TTSEngine
from .kokoro import KokoroEngine


def create_tts_engine(engine: str = "kokoro", voice: Optional[str] = None, **kwargs) -> TTSEngine:
    """
    Create a TTS engine instance.
    
    Args:
        engine: Engine name - "kokoro" or "qwen3"
        voice: Voice name (optional, uses default if not specified)
        **kwargs: Additional arguments passed to the engine constructor
    
    Returns:
        TTSEngine instance
    
    Examples:
        # Create Kokoro engine (default)
        engine = create_tts_engine("kokoro")
        
        # Create Kokoro with specific voice
        engine = create_tts_engine("kokoro", voice="am_michael")
        
        # Create Qwen3 engine (requires GPU)
        engine = create_tts_engine("qwen3", voice="Serena")
    """
    engine = engine.lower()
    
    if engine == "kokoro":
        # Use default voice if not specified
        if voice is None:
            voice = "af_sarah"
        return KokoroEngine(voice=voice, **kwargs)
    elif engine == "qwen3":
        from .qwen import Qwen3Engine

        # Use default voice if not specified
        if voice is None:
            voice = "Ryan"
        return Qwen3Engine(voice=voice, **kwargs)
    else:
        raise ValueError(f"Unknown engine: {engine}. Available: kokoro, qwen3")


__all__ = [
    "TTSEngine",
    "KokoroEngine", 
    "create_tts_engine"
]
