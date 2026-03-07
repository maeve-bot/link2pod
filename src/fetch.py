"""
Web page fetching module.
"""

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Configuration
DEFAULT_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Tags to remove from HTML
SKIP_TAGS = [
    "script", "style", "meta", "head", "link", "noscript", 
    "nav", "header", "footer", "aside", "form", "button",
    "iframe", "svg", "canvas", "video", "audio", "embed", "object"
]

# Content-bearing tags to extract
CONTENT_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "article", "section", "blockquote"]


def fetch_webpage(url: str, max_length: int = 10000, use_playwright: bool = False) -> str:
    """
    Fetch a web page and extract its main text content.
    
    Args:
        url: The URL to fetch
        max_length: Maximum characters to return (default: 10000)
        use_playwright: If True, use Playwright for JS-rendered pages (default: False)
    
    Returns:
        Extracted text content from the page
    """
    if use_playwright:
        return fetch_with_playwright(url, max_length)
    else:
        return fetch_with_requests(url, max_length)


def fetch_with_requests(url: str, max_length: int = 10000) -> str:
    """Fetch using requests (fast, works for static pages)."""
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=DEFAULT_TIMEOUT,
        allow_redirects=True
    )
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, "lxml")
    
    # Remove unwanted elements
    for tag in soup(SKIP_TAGS):
        tag.decompose()
    
    # Try to find main content areas
    content = extract_main_content(soup)
    
    # Truncate if needed
    content = truncate_content(content, max_length)
    
    return content


def fetch_with_playwright(url: str, max_length: int = 10000) -> str:
    """
    Fetch using Playwright (for JavaScript-rendered pages).
    
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright not installed. Install with:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
    
    with sync_playwright() as p:
        # Launch browser (headless by default)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate and wait for content to load
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        # Wait for page to be fully loaded
        page.wait_for_load_state("networkidle", timeout=15000)
        
        # Extended wait for JS-heavy sites
        page.wait_for_timeout(5000)
        
        # Scroll in increments to trigger lazy loading
        for i in range(3):
            page.evaluate(f"window.scrollTo(0, {i * 1000})")
            page.wait_for_timeout(1000)
        
        # Scroll back to top
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        
        # Get the rendered HTML
        content = page.content()
        browser.close()
    
    # Parse the rendered HTML
    soup = BeautifulSoup(content, "lxml")
    
    # Remove unwanted elements
    for tag in soup(SKIP_TAGS):
        tag.decompose()
    
    # Extract content
    content = extract_main_content(soup)
    
    # If still too little, try getting body text directly
    if len(content) < 500:
        body = soup.find("body")
        if body:
            content = extract_text_from_element(body)
    
    # Truncate if needed
    content = truncate_content(content, max_length)
    
    return content


def truncate_content(content: str, max_length: int) -> str:
    """Truncate content to max_length at sentence boundary."""
    if len(content) > max_length:
        content = content[:max_length]
        # Try to end at a sentence boundary
        last_period = content.rfind('.')
        last_exclaim = content.rfind('!')
        last_question = content.rfind('?')
        end = max(last_period, last_exclaim, last_question)
        if end > max_length * 0.8:  # Only truncate if we're near the end
            content = content[:end + 1]
        content += "..."
    
    return content


def extract_main_content(soup: BeautifulSoup) -> str:
    """
    Extract the main content from a web page.
    
    Tries multiple strategies:
    1. Look for <article> tag
    2. Look for <main> tag
    3. Look for common content class names (article, content, post, entry, etc.)
    4. Fall back to extracting all paragraphs
    """
    # Strategy 1: <article> tag
    article = soup.find("article")
    if article:
        return extract_text_from_element(article)
    
    # Strategy 2: <main> tag
    main = soup.find("main")
    if main:
        return extract_text_from_element(main)
    
    # Strategy 3: Look for content-containing elements
    for selector in [
        "[class*='article']", 
        "[class*='content']", 
        "[class*='post']",
        "[class*='entry']",
        "[id*='article']",
        "[id*='content']",
        "[id*='post']",
    ]:
        elem = soup.select_one(selector)
        if elem:
            text = extract_text_from_element(elem)
            if len(text) > 200:  # Only use if substantial content
                return text
    
    # Strategy 4: Extract all paragraph text
    return extract_text_from_element(soup)


def extract_text_from_element(element: BeautifulSoup) -> str:
    """Extract clean text from a BeautifulSoup element."""
    texts = []
    
    for tag in element.find_all(CONTENT_TAGS):
        text = tag.get_text(separator=' ', strip=True)
        if text and len(text) > 20:  # Skip very short fragments
            texts.append(text)
    
    # If nothing found with specific tags, get all text
    if not texts:
        text = element.get_text(separator=' ', strip=True)
        if text:
            texts = [text]
    
    # Join with double newlines between paragraphs
    return '\n\n'.join(texts)


def get_page_title(soup: BeautifulSoup, url: str) -> str:
    """Extract the page title."""
    # Try <title> tag
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    
    # Try <h1> tag
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    
    # Fall back to URL
    from urllib.parse import urlparse
    return urlparse(url).netloc


def fetch_local_file(file_path: str, max_length: int = 10000) -> str:
    """
    Read content from a local file (txt, md, or other text files).
    
    Args:
        file_path: Path to the local file
        max_length: Maximum characters to return (default: 10000)
    
    Returns:
        File contents as string
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    
    # Read the file
    content = path.read_text(encoding='utf-8')
    
    # Apply max_length truncation
    if len(content) > max_length:
        content = content[:max_length]
        content += "..."
    
    return content


def is_url(string: str) -> bool:
    """Check whether a string is a valid HTTP(S) URL."""
    parsed = urlparse(string)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
