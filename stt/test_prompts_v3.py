#!/usr/bin/env python3
"""
Script iteration testing v3 - refined prompt.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import get_openrouter_api_key, OPENROUTER_API_URL
import requests


# V3 PROMPT - Refined with better path handling
PODCAST_SYSTEM_PROMPT_V3 = """You are transforming text content into an audiobook-style script for spoken audio. Your goal is to preserve ALL information while making it pleasant to listen to.

## CORE PRINCIPLE

Keep EVERY piece of content. This is an audiobook, not a summary. Don't drop anything.

## FORMATTING RULES

- Write plain text only, no markdown, no asterisks
- Use pause tags sparingly: {SHORT} (0.5s), {MED} (1s), {LONG} (2s)

## PROBLEM ELEMENT HANDLING

### File Paths
Transform for natural speech:
- Unix paths: "/var/lib/docker" → "var/lib/docker" (just skip leading slash)
- Spell "etc", "var", "lib" as words, not letters
- "C:\ProgramData\docker" → "C:\\ProgramData\\docker" or "C drive ProgramData docker"

### Code/Commands
Describe what the command does, keep it recognizable:
- "docker run -p 8080:80 nginx" → "docker run with port 8080 mapped to 80, for the nginx image"
- "docker network create my-network" → "docker network create, naming it my-network"

### Numbers and Ports
- "8080:80" → "port 8080 to port 80" or "port 8080, colon, port 80"
- Speak version numbers naturally: "version 1.2.3"

### Bullets/Lists
Convert to flowing prose: "First... Second... Finally..."

### URLs
- Say "link" or describe: "the Docker docs link" or "documentation at docs.docker.com"

### Section Headers  
Transform to verbal: "## Key Concepts" → "Now for the key concepts"

## VOICE

Natural, like explaining to a knowledgeable colleague. No hedging.
"""


def test_prompt_version(content: str, source_name: str, prompt_version: str = "v3", model: str = "openai/gpt-4o-mini") -> str:
    """Test a specific prompt version."""
    api_key = get_openrouter_api_key()
    
    if prompt_version == "v3":
        system_prompt = PODCAST_SYSTEM_PROMPT_V3
    elif prompt_version == "v2":
        from stt.test_prompts import PODCAST_SYSTEM_PROMPT_V2
        system_prompt = PODCAST_SYSTEM_PROMPT_V2
    else:
        from src.llm import PODCAST_SYSTEM_PROMPT
        system_prompt = PODCAST_SYSTEM_PROMPT
    
    user_prompt = f"""Transform for audio. Keep ALL content - this is an audiobook, not a summary.

SOURCE: {source_name}

CONTENT:
---
{content}
---

Make it speakable while preserving every fact, command, and detail."""

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
        "temperature": 0.5
    }
    
    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise RuntimeError(f"API error: {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def main():
    test_file = Path(__file__).parent.parent / "test_content_docker.txt"
    content = test_file.read_text()
    
    print("=== TESTING V3 PROMPT ===\n")
    result = test_prompt_version(content, "test_content_docker.txt", "v3")
    print(result)
    
    output_path = Path(__file__).parent.parent / "output" / "test_v3_script.txt"
    output_path.write_text(result)
    print(f"\n=== Saved to {output_path} ===")


if __name__ == "__main__":
    main()
