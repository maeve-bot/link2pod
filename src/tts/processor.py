"""
TTS transcript pre-processor.
Handles pause tags from LLM output and applies them during synthesis.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Pause durations based on split level (in milliseconds)
PARAGRAPH_PAUSE_MS = 1000
SENTENCE_PAUSE_MS = 800
WORD_PAUSE_MS = 500

# Default pause if no tags present (in milliseconds)
DEFAULT_PAUSE_MS = 500


@dataclass
class TextChunk:
    """A text chunk with pause info based on what split level created it."""
    text: str
    pause_ms: int = 0


# Split levels for pause determination
class SplitLevel:
    PARAGRAPH = "paragraph"  # Double newline split
    SENTENCE = "sentence"    # .!? split
    WORD = "word"            # Force-split at max_size

# Pause duration by split level
SPLIT_LEVEL_PAUSES = {
    SplitLevel.PARAGRAPH: PARAGRAPH_PAUSE_MS,
    SplitLevel.SENTENCE: SENTENCE_PAUSE_MS,
    SplitLevel.WORD: WORD_PAUSE_MS,
}


def parse_transcript_with_pauses(transcript: str) -> list[TextChunk]:
    """
    Parse transcript and extract pause tags.
    
    Note: Legacy function - pause tags are deprecated in favor of 
    automatic pause determination based on chunking level.
    
    Args:
        transcript: Raw transcript with {SHORT}, {MED}, {LONG} tags (deprecated)
    
    Returns:
        List of TextChunk objects with text and pause durations
    """
    # For backward compatibility, still handle tags but don't require them
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
        
        # Get the pause duration (legacy support)
        tag = match.group(1)
        legacy_pauses = {"{SHORT}": 500, "{MED}": 1000, "{LONG}": 2000}
        pause_ms = legacy_pauses.get(tag, 0)
        
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