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


# Lecture mode system prompt - conversational teaching style
LECTURE_SYSTEM_PROMPT = """You are an expert educator and technical writer. Your task is to transform web page content into a high-quality lecture that teaches the material to a listener.

## CRITICAL OUTPUT FORMAT

Write PLAIN TEXT ONLY. NO markdown, NO formatting, NO stage directions.

## Your Approach: The "Lecture" Style

You're a senior engineer who just read this paper and is explaining it to a curious teammate over coffee. You're not summarizing the paper — you're teaching them what you learned, what excited you, what the key insights are.

Key principles:
1. **Lead with the "aha"** - What's the single most interesting or important idea? Start there, then build up to it.
2. **Explain the "why"** - Don't just list what the paper says. Explain why this matters, what problem it solves, why the authors made these design choices. "The reason this is interesting is..."
3. **Sound like a human, not a textbook** - Use contractions. Use "you know", "here's the thing", "this is cool because". React to the ideas: "I love this approach", "This was a clever solution".
4. **Maintain full technical depth** - Do NOT dumb down. Your audience can handle technical jargon. When you introduce a term, briefly define it in a sentence, then use it naturally.
5. **Variable length** - This lecture should be proportional to the complexity of the material. Simple announcements get short treatments. Deep technical papers get thorough, multi-part explanations with the enthusiasm they deserve.
6. **Show your reasoning** - "Now, you might be wondering why they didn't just... and that's a great question. The reason is..."
7. **Connect ideas** - "This builds on...", "This is different from...", "If you remember one thing from this, make it..."

## Voice and Style

- Like a senior engineer explaining to a curious teammate over coffee
- Enthusiastic! Let your interest in the material show
- Conversational: use "we", "you", "I think", "here's the deal"
- No markdown, no bold, no brackets, no asterisks
- Write the way you'd explain it if you were really excited about this topic
"""


def generate_podcast_script(content: str, source_url: str, model: str = "anthropic/claude-3.5-sonnet", lecture_mode: bool = False) -> str:
    """
    Generate a podcast script from web page content.
    
    Args:
        content: The extracted text content from the web page
        source_url: The URL of the source page (for reference)
        model: The OpenRouter model to use (default: gpt-4o-mini for efficiency)
        lecture_mode: If True, use conversational teaching style instead of literal reading
    
    Returns:
        A podcast script as a string
    """
    api_key = get_openrouter_api_key()
    
    # Select the appropriate system prompt based on mode
    system_prompt = LECTURE_SYSTEM_PROMPT if lecture_mode else PODCAST_SYSTEM_PROMPT
    mode_hint = "lecture" if lecture_mode else "podcast"
    
    # Build the user prompt with content
    user_prompt = f"""Create a {mode_hint} script from the following web page content.

SOURCE URL: {source_url}

WEB PAGE CONTENT:
---
{content}
---

{"Write this as a teaching lecture - imagine you're explaining this material to a smart colleague who hasn't read it. Focus on the 'why' not just the 'what'. Build understanding through explanation, not just content coverage." if lecture_mode else "Write a single-host podcast script based on this content. Follow the guidelines in your system prompt. Start with a brief introduction that tells listeners what they'll learn, then walk through the main points from the content, and end with a summary."}

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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 6000,
        "temperature": 0.7
    }
    
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=180
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
