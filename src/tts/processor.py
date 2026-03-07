"""
TTS transcript pre-processor.
Handles pause tags from LLM output and applies them during synthesis.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Pause durations in milliseconds
PAUSE_DURATIONS = {
    "{SHORT}": 500,
    "{MED}": 1000,
    "{LONG}": 2000,
}


@dataclass
class TextChunk:
    """A text chunk with optional pause after it."""
    text: str
    pause_ms: int = 0


def parse_transcript_with_pauses(transcript: str) -> list[TextChunk]:
    """
    Parse transcript and extract pause tags.
    
    Args:
        transcript: Raw transcript with {SHORT}, {MED}, {LONG} tags
    
    Returns:
        List of TextChunk objects with text and pause durations
    """
    # Normalize pause tags - replace with unique placeholder
    processed = transcript
    
    # Find all pause tags and their positions
    # Use regex to find each tag and what comes after it
    pattern = r'(\{SHORT\}|\{MED\}|\{LONG\})'
    
    chunks = []
    last_end = 0
    
    for match in re.finditer(pattern, processed):
        # Get text before this tag
        before_text = processed[last_end:match.start()].strip()
        
        if before_text:
            chunks.append(TextChunk(text=before_text, pause_ms=0))
        
        # Get the pause duration
        tag = match.group(1)
        pause_ms = PAUSE_DURATIONS.get(tag, 0)
        
        # Add a placeholder chunk for the pause (empty text)
        chunks.append(TextChunk(text="", pause_ms=pause_ms))
        
        last_end = match.end()
    
    # Get remaining text after last tag
    remaining = processed[last_end:].strip()
    if remaining:
        chunks.append(TextChunk(text=remaining, pause_ms=0))
    
    # Filter out empty chunks (except those with pauses)
    return [c for c in chunks if c.text or c.pause_ms]


def strip_pause_tags(text: str) -> str:
    """Remove all pause tags from text (for plain text output)."""
    return re.sub(r'\{SHORT\}|\{MED\}|\{LONG\}', '', text)


# Default pause if no tags present (in milliseconds)
DEFAULT_PAUSE_MS = 500