#!/usr/bin/env python3
"""
Script iteration testing v4 - handling nginx and other pronunciation issues.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import get_openrouter_api_key, OPENROUTER_API_URL
import requests


# V4 PROMPT - Handles pronunciation-challenged words
PODCAST_SYSTEM_PROMPT_V4 = """Transform text into an audiobook-style script for spoken audio. Preserve ALL content while making it speakable.

## RULES

- Plain text only, no markdown
- Pause tags: {SHORT} {MED} {LONG}

## PRONUNCIATION FIXES

These common tech terms are hard for TTS to pronounce. Transform them:

- "nginx" → "the nginx web server" or "N-G-I-N-X"
- "Docker" → "Docker" (correct)
- "docker0" → "docker zero" (the network name, spoken as two words)
- URLs → describe: "the Docker docs link" or spell domain: "D-O-C-S dot D-O-C-K-E-R dot C-O-M"

## PATH HANDLING

- Unix paths: skip leading slash, speak naturally: "var/lib/docker/network/files"
- Windows paths: "C:\\path" → "C drive path" or spell: "C colon backslash path"

## CODE/COMMANDS

Transform to speakable descriptions:
- "docker run -p 8080:80 nginx" → "docker run with port 8080 to port 80 for the nginx web server"
- "docker network create my-network" → "docker network create, naming it my-network"

## LIST CONTENT

Convert bullets to flowing prose with "First...", "Second...", etc.

## VOICE

Natural, expert explanation. Preserve EVERY fact.
"""


def test_prompt(content: str, source_name: str, prompt: str, model: str = "openai/gpt-4o-mini") -> str:
    api_key = get_openrouter_api_key()
    
    user_prompt = f"""Transform for audio. Keep ALL content.

SOURCE: {source_name}

CONTENT:
---
{content}
---

Make speakable. Use pause tags only for natural breaks."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.ai",
        "X-Title": "Link2Pod"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
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
    
    print("=== TESTING V4 PROMPT ===\n")
    result = test_prompt(content, "test_content_docker.txt", PODCAST_SYSTEM_PROMPT_V4)
    print(result)
    
    output_path = Path(__file__).parent.parent / "output" / "test_v4_script.txt"
    output_path.write_text(result)
    print(f"\n=== Saved to {output_path} ===")


if __name__ == "__main__":
    main()
