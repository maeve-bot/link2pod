"""
TTS transcript processor.
Pause durations determined automatically by chunking level.
"""

# Pause durations based on split level (in milliseconds)
PARAGRAPH_PAUSE_MS = 1000
SENTENCE_PAUSE_MS = 800
WORD_PAUSE_MS = 500

# Default pause (used if no split level available)
DEFAULT_PAUSE_MS = 500


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
