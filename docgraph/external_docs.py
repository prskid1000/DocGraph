"""External documentation crawling and indexing.

Separate phase from wiki generation:
1. Crawl URLs → detect content type → extract content
2. Store indexed docs in memory/file cache
3. Wiki builder reads indexed docs as context

This allows:
- Crawl once, reuse across wiki regenerations
- Disable crawling after first run
- Separate progress tracking (crawl phase vs wiki phase)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from docgraph.wiki_crawl import crawl_urls, CrawledPage
    HAVE_CRAWL = True
except ImportError:
    HAVE_CRAWL = False


@dataclass
class IndexedExternalDoc:
    """An indexed external documentation page."""
    url: str
    title: str
    content: str
    content_type: str  # "code" | "html" | "text"
    depth: int
    indexed_at: float


EXTERNAL_DOCS_CACHE_FILE = ".docgraph/external-docs-cache.json"


def _cache_path(cfg) -> Path:
    """Return path to external docs cache file."""
    return Path(cfg.data_dir) / EXTERNAL_DOCS_CACHE_FILE


def save_indexed_docs(cfg, docs: list[IndexedExternalDoc]) -> None:
    """Save crawled and indexed docs to disk cache."""
    try:
        path = _cache_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(d) for d in docs]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.debug(f"Saved {len(docs)} indexed external docs to {path}")
    except Exception as e:
        log.warning(f"Failed to save indexed external docs: {e}")


def load_indexed_docs(cfg) -> list[IndexedExternalDoc]:
    """Load cached indexed external docs from disk."""
    try:
        path = _cache_path(cfg)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        docs = [IndexedExternalDoc(**d) for d in payload]
        log.debug(f"Loaded {len(docs)} indexed external docs from cache")
        return docs
    except Exception as e:
        log.warning(f"Failed to load indexed external docs: {e}")
        return []


async def crawl_and_index_external_docs(
    start_urls: list[str],
    max_depth: int = 2,
    exclude_patterns: list[str] | None = None,
    progress_cb: callable | None = None,
) -> list[IndexedExternalDoc]:
    """Crawl URLs and return indexed documents.
    
    This is a separate phase from wiki generation. Results can be cached
    and reused across multiple wiki builds.
    
    Args:
        start_urls: List of URLs to crawl (e.g., docs.example.com)
        max_depth: Max crawl depth (1-5)
        exclude_patterns: URL patterns to skip
        progress_cb: Callback for progress updates
    
    Returns:
        List of IndexedExternalDoc objects
    """
    if not HAVE_CRAWL:
        log.warning("crawl_and_index: aiohttp/BeautifulSoup not available")
        return []
    
    if not start_urls:
        return []
    
    indexed: list[IndexedExternalDoc] = []
    
    try:
        import time
        for url_idx, url in enumerate(start_urls):
            if progress_cb:
                try:
                    progress_cb("crawl", url_idx, len(start_urls), url[:60])
                except Exception:
                    pass
            
            try:
                crawled = await asyncio.wait_for(
                    crawl_urls(
                        url,
                        max_depth=max_depth,
                        exclude_patterns=exclude_patterns or [],
                    ),
                    timeout=120,
                )
                
                for page in crawled:
                    doc = IndexedExternalDoc(
                        url=page.url,
                        title=page.title,
                        content=page.content,
                        content_type=page.content_type,
                        depth=page.depth,
                        indexed_at=time.time(),
                    )
                    indexed.append(doc)
            except asyncio.TimeoutError:
                log.warning(f"crawl_and_index: timeout crawling {url}")
            except Exception as e:
                log.warning(f"crawl_and_index: failed to crawl {url}: {e}")
        
        if progress_cb and indexed:
            try:
                progress_cb("index", len(indexed), len(indexed), "")
            except Exception:
                pass
    
    except Exception as e:
        log.error(f"crawl_and_index: overall failure: {e}")
    
    return indexed


def format_indexed_docs_for_prompt(docs: list[IndexedExternalDoc], limit: int = 5) -> str:
    """Format indexed docs into a prompt snippet for wiki generation.
    
    Limits to top `limit` docs by depth (shallowest first, most relevant).
    """
    if not docs:
        return ""
    
    # Sort by depth (shallowest = most relevant) and take top N
    sorted_docs = sorted(docs, key=lambda d: (d.depth, d.url))[:limit]
    
    lines = ["## External Documentation Context\n"]
    for doc in sorted_docs:
        lines.append(f"### {doc.title or doc.url}\n")
        lines.append(f"**URL:** {doc.url}\n")
        lines.append(f"**Type:** {doc.content_type}\n")
        # Limit content to 1500 chars per doc
        lines.append(f"\n{doc.content[:1500]}\n")
        lines.append("---\n")
    
    return "\n".join(lines)


def clear_external_docs_cache(cfg) -> None:
    """Clear the cached external docs."""
    try:
        path = _cache_path(cfg)
        if path.exists():
            path.unlink()
            log.debug("Cleared external docs cache")
    except Exception as e:
        log.warning(f"Failed to clear external docs cache: {e}")
