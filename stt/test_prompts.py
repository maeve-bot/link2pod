#!/usr/bin/env python3
"""
Script iteration testing - compare different prompts without TTS.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import get_openrouter_api_key, OPENROUTER_API_URL
import requests


# V2 PROMPT - Audiobook style with explicit problem handling
PODCAST_SYSTEM_PROMPT_V2 = """You are transforming text content into an audiobook-style script that will be spoken aloud. Your ONLY job is to make the content pleasant to listen to while preserving EVERY piece of information from the source.

## ABSOLUTE RULES

1. **KEEP ALL CONTENT** - Every fact, example, command, and detail must be included. Never summarize. Never condense. Never drop "less important" bits.

2. **WRITE PLAIN TEXT ONLY** - No markdown, no asterisks, no brackets, no stage directions.

3. **USE PAUSE TAGS SPARINGLY** - Only for natural breaks in thought:
   - {SHORT} - Brief pause (0.5s)
   - {MED} - Medium pause (1s)
   - {LONG} - Longer pause (2s)

## HOW TO HANDLE PROBLEMATIC CONTENT

### File Paths and URLs
- Transform into spoken descriptions:
  - "/var/lib/docker" → "the Docker library directory, located at V-A-R slash lib slash docker"
  - "https://docs.docker.com" → "the Docker documentation website"
- OR spell them out letter-by-letter if they're essential:
  - "Command: docker run -p 8080:80" → "Run the command docker run, specifying port 8080 mapped to port 80"

### Code Blocks and Commands
- Describe what the code does INSTEAD of reading it verbatim:
  - "docker network create my-network" → "create a network called my network using the Docker network create command"
- Keep technical accuracy but make it speakable

### Bullet Points and Lists
- Convert to flowing prose with transitions:
  - "1. First item" → "First," 
  - "2. Second item" → "Second,"
- Use "first," "second," "finally," etc. as connectors

### Numbers in Technical Contexts
- Port numbers: "8080:80" → "port 8080 to port 80"
- Version numbers: "v1.2.3" → "version 1.2.3"
- Paths with numbers: Spell out when ambiguous

### Section Headers
- Turn into verbal transitions: "## Configuration" → "Now let's look at configuration"

## VOICE

- Natural, like someone explaining to a colleague
- Confident, no hedging ("maybe", "I think")
- No emoji, no slang unless source has it
"""


def test_prompt_version(content: str, source_name: str, prompt_version: str = "v2", model: str = "openai/gpt-4o-mini") -> str:
    """Test a specific prompt version."""
    api_key = get_openrouter_api_key()
    
    if prompt_version == "v2":
        system_prompt = PODCAST_SYSTEM_PROMPT_V2
    else:
        from src.llm import PODCAST_SYSTEM_PROMPT
        system_prompt = PODCAST_SYSTEM_PROMPT
    
    user_prompt = f"""Transform the following content into an audiobook-style script for audio output.

SOURCE: {source_name}

CONTENT:
---
{content}
---

Follow the rules in your system prompt. Include ALL information from the source. Make it pleasant to listen to."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.ai",
        "X-Title": "Link2Pod"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4000,
        "temperature": 0.5  # Lower temperature for consistency
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise RuntimeError(f"API error: {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def main():
    test_file = Path(__file__).parent.parent / "test_content_docker.txt"
    content = test_file.read_text()
    
    print("=== TESTING V2 PROMPT ===\n")
    result = test_prompt_version(content, "test_content_docker.txt", "v2")
    print(result)
    
    # Save for comparison
    output_path = Path(__file__).parent.parent / "output" / "test_v2_script.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(result)
    print(f"\n=== Saved to {output_path} ===")


if __name__ == "__main__":
    main()
