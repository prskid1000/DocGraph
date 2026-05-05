"""URL crawling and content detection for wiki building.

Handles external documentation/website discovery and parsing.
- detect_content_type: Determine if URL is code docs, website, or other
- crawl_urls: Discover linked pages up to max_depth
- parse_doc_page: Extract content from code documentation
- parse_html_page: Extract main content from HTML pages
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
from collections import deque

log = logging.getLogger(__name__)

try:
    import aiohttp
    from bs4 import BeautifulSoup
    HAVE_CRAWL_DEPS = True
except ImportError:
    HAVE_CRAWL_DEPS = False


@dataclass
class CrawledPage:
    """Represents a fetched page from a URL."""
    url: str
    title: str
    content: str  # Main content (text)
    content_type: str  # "code" | "html" | "text" | "other"
    depth: int  # How deep in the crawl tree


def _detect_content_type(
    url: str,
    content_type_header: str,
    sample_html: str,
) -> str:
    """Determine content type based on URL, headers, and sample content.
    
    Returns: "code" | "html" | "text" | "pdf" | "other"
    """
    # URL-based heuristics
    url_lower = url.lower()
    if any(x in url_lower for x in (
        "github.com", "gitlab.com", "bitbucket.org",
        "docs.python.org", "golang.org", "docs.rs",
        "/api/docs", "/api/reference", "/docs/",
    )):
        return "code"
    
    # Content-Type header
    ct = content_type_header.lower() if content_type_header else ""
    if "text/plain" in ct or "text/markdown" in ct:
        return "text"
    if "application/pdf" in ct or url_lower.endswith(".pdf"):
        return "pdf"
    if "text/html" in ct:
        return "html"
    
    # Sample content heuristics
    if sample_html:
        sample_lower = sample_html[:2000].lower()
        # Code-like: <code>, <pre>, syntax highlighting
        if any(x in sample_lower for x in ("<code", "<pre", "highlight", "language-")):
            return "code"
        # API docs: OpenAPI, Swagger, API reference markers
        if any(x in sample_lower for x in ("openapi", "swagger", "api reference", "endpoint", "parameter")):
            return "code"
        # Regular HTML
        if "<html" in sample_lower or "<body" in sample_lower:
            return "html"
    
    return "other"


def _is_same_domain(base_url: str, target_url: str) -> bool:
    """Check if target_url is on the same domain as base_url."""
    base_parsed = urlparse(base_url)
    target_parsed = urlparse(target_url)
    return base_parsed.netloc == target_parsed.netloc


def _should_exclude_url(
    url: str,
    exclude_patterns: list[str],
) -> bool:
    """Check if URL matches any exclusion pattern."""
    if not exclude_patterns:
        return False
    url_lower = url.lower()
    for pattern in exclude_patterns:
        pattern_lower = pattern.lower()
        # Simple glob/regex matching
        if "*" in pattern_lower:
            # Convert glob to regex
            regex = pattern_lower.replace(".", r"\.").replace("*", ".*")
            if re.search(regex, url_lower):
                return True
        elif pattern_lower in url_lower:
            return True
    return False


def _extract_title_from_html(html: str) -> str:
    """Extract page title from HTML."""
    if not HAVE_CRAWL_DEPS:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        if title:
            return title.get_text().strip()[:100]
    except Exception:
        pass
    return ""


def _extract_main_content_html(html: str) -> str:
    """Extract main content from HTML page (removes nav, footer, etc)."""
    if not HAVE_CRAWL_DEPS:
        return html[:2000]
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script, style, nav, footer
        for tag in soup.find_all(("script", "style", "nav", "footer", "aside")):
            tag.decompose()
        # Try to find main content area
        main = soup.find(("main", "article", "section")) or soup.find("div", class_=re.compile(r"content|body|article", re.I))
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        return text[:5000]  # Cap at 5K chars
    except Exception:
        # Fallback: strip basic HTML tags
        text = re.sub(r"<[^>]+>", "\n", html)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text[:5000]


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract all <a href=...> links from HTML."""
    if not HAVE_CRAWL_DEPS:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            # Convert relative URLs to absolute
            url = urljoin(base_url, href)
            # Remove fragments
            url = url.split("#")[0]
            if url not in links:
                links.append(url)
        return links[:20]  # Cap at 20 links per page
    except Exception:
        return []


async def crawl_urls(
    start_url: str,
    max_depth: int = 2,
    exclude_patterns: list[str] | None = None,
    timeout_sec: int = 30,
    progress_cb: callable | None = None,
) -> list[CrawledPage]:
    """Crawl a starting URL and discover linked pages up to max_depth.
    
    Returns a list of CrawledPage objects. Respects domain boundaries
    and exclusion patterns. Single-threaded but async-io based.
    """
    if not HAVE_CRAWL_DEPS:
        log.warning("crawl_urls: aiohttp/BeautifulSoup not available; skipping crawl")
        return []
    
    exclude_patterns = exclude_patterns or []
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    visited = set()
    results: list[CrawledPage] = []
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    
    async def _fetch(url: str) -> Optional[tuple[str, str]]:
        """Fetch URL, return (content, content_type_header) or None."""
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status != 200:
                        return None
                    ct = resp.headers.get("content-type", "")
                    content = await resp.text()
                    return content, ct
        except Exception as e:
            log.debug(f"crawl_urls: fetch {url} failed: {e}")
            return None
    
    while queue:
        url, depth = queue.popleft()
        
        # Skip visited, too deep, excluded, or different domain
        if url in visited or depth > max_depth:
            continue
        if not _is_same_domain(start_url, url):
            continue
        if _should_exclude_url(url, exclude_patterns):
            continue
        
        visited.add(url)
        
        if progress_cb:
            try:
                progress_cb("crawl", len(visited), max_depth, url[:80])
            except Exception:
                pass
        
        # Fetch the URL
        fetch_result = await _fetch(url)
        if not fetch_result:
            continue
        
        content, ct = fetch_result
        if not content:
            continue
        
        # Detect type
        content_type = _detect_content_type(url, ct, content[:2000])
        
        # Extract title
        title = _extract_title_from_html(content) if content_type == "html" else url
        
        # Extract main content
        if content_type == "html":
            main_content = _extract_main_content_html(content)
        else:
            # For text/code, take first 5K
            main_content = content[:5000]
        
        # Store result
        page = CrawledPage(
            url=url,
            title=title,
            content=main_content,
            content_type=content_type,
            depth=depth,
        )
        results.append(page)
        
        # Extract and queue new links if not too deep
        if depth < max_depth and content_type == "html":
            links = _extract_links_from_html(content, url)
            for link in links:
                if link not in visited:
                    queue.append((link, depth + 1))
    
    return results


async def crawl_urls_sync(
    start_url: str,
    max_depth: int = 2,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """Synchronous wrapper for crawl_urls. Returns list of dicts."""
    pages = await crawl_urls(start_url, max_depth, exclude_patterns)
    return [
        {
            "url": p.url,
            "title": p.title,
            "content": p.content,
            "type": p.content_type,
            "depth": p.depth,
        }
        for p in pages
    ]


def format_crawled_content(pages: list[CrawledPage]) -> str:
    """Format crawled pages into a single markdown-like text block for prompting."""
    if not pages:
        return ""
    
    lines = ["## Crawled External Documentation\n"]
    for p in pages:
        lines.append(f"### {p.title or p.url}\n")
        lines.append(f"**URL:** {p.url}\n")
        lines.append(f"**Type:** {p.content_type}\n")
        lines.append(f"**Depth:** {p.depth}\n")
        lines.append(f"\n{p.content[:1500]}\n")
        lines.append("---\n")
    
    return "\n".join(lines)
