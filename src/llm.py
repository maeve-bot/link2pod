"""
LLM processing module - converts web page content into a podcast script.
"""

import os
from typing import Optional

import requests

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load API key from environment
def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
        )
    
    return api_key


# System prompt for podcast script generation
PODCAST_SYSTEM_PROMPT = """You are a professional podcast script writer. Your task is to transform web page content into an engaging, listenable podcast script read by a single host.

## CRITICAL OUTPUT FORMAT

Write PLAIN TEXT ONLY. NO markdown, NO formatting, NO stage directions.

Use these pause tags to control pacing:
- {SHORT} - A brief pause (0.5 seconds)
- {MED} - A medium pause (1 second)  
- {LONG} - A significant pause (2 seconds)

Example of CORRECT output:
"Welcome to today's episode. {SHORT} We're going to talk about game development. {MED} First, let's discuss the basics. {LONG} Now let's move on to advanced topics."

Example of WRONG output (do NOT do this):
"**[Pause]** Welcome to today's episode... *pause* ...We're going to talk about..."
"**Host:** Welcome to today's episode..."

## Your Approach

1. **Stay faithful to the source** - Only present information that is actually in the provided content. Don't add facts, opinions, or commentary that aren't supported by the source material.

2. **Structure for listening** - Use verbal transitions ("Now let's move on to...", "Next up is...") instead of formatting.

3. **Hook the listener** - Start with an engaging intro that summarizes what they'll learn.

4. **End with a summary** - Wrap up the main points so listeners walk away with key takeaways.

## Voice and Style

- Professional but approachable
- Enthusiastic about the topic
- Plain text conversational language
- No markdown, no bold, no brackets, no asterisks
"""


def generate_podcast_script(content: str, source_url: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Generate a podcast script from web page content.
    
    Args:
        content: The extracted text content from the web page
        source_url: The URL of the source page (for reference)
        model: The OpenRouter model to use (default: gpt-4o-mini for efficiency)
    
    Returns:
        A podcast script as a string
    """
    api_key = get_openrouter_api_key()
    
    # Build the user prompt with content
    user_prompt = f"""Create a podcast script from the following web page content.

SOURCE URL: {source_url}

WEB PAGE CONTENT:
---
{content}
---

Write a single-host podcast script based on this content. Follow the guidelines in your system prompt. Start with a brief introduction that tells listeners what they'll learn, then walk through the main points from the content, and end with a summary.

Remember: Only use information from the provided content. Don't add external facts or opinions."""

    # Make API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.ai",
        "X-Title": "Link2Pod"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PODCAST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4000,
        "temperature": 0.7
    }
    
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter API error (status {response.status_code}): {response.text}"
        )
    
    try:
        result = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"OpenRouter API returned invalid JSON (status {response.status_code}): {response.text}"
        ) from e
    
    if "choices" not in result or len(result["choices"]) == 0:
        raise ValueError(f"Unexpected API response: {result}")
    
    script = result["choices"][0]["message"]["content"]
    
    return script.strip()


def generate_podcast_script_with_metadata(
    content: str, 
    source_url: str, 
    title: Optional[str] = None,
    model: str = "openai/gpt-4o-mini"
) -> dict:
    """
    Generate a podcast script with metadata.
    
    Returns a dict with:
    - script: The podcast script text
    - title: Suggested podcast title
    - duration_estimate: Estimated audio duration in seconds
    """
    script = generate_podcast_script(content, source_url, model)
    
    # Estimate duration (average speaking rate ~150 words/min)
    word_count = len(script.split())
    duration_estimate = int((word_count / 150) * 60)
    
    return {
        "script": script,
        "title": title or f"Podcast: {source_url}",
        "duration_estimate": duration_estimate,
        "word_count": word_count
    }
