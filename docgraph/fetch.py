"""Web crawler for external-link pre-fetch.

Downloads URLs to `<root>/.docgraph/external/` as `.html` files that
docgraph's tree-sitter HTML parser indexes alongside source code.
Uses a depth-limited BFS (same-domain only) and a TTL gate so re-runs
skip URLs that are still fresh.

No aiohttp dependency — stdlib urllib only so this stays importable even
in the slimmest docgraph install.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

from docgraph.links import ExternalLink, load_links, save_links

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "docgraph-fetch/1.0 (local knowledge-graph indexer)",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en",
}
_TIMEOUT = 20
_MAX_BYTES = 2_000_000   # 2 MB per page; larger pages are truncated


def _url_slug(url: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", url.rstrip("/"))
    if len(safe) > 80:
        safe = safe[:40] + "_" + hashlib.sha1(url.encode()).hexdigest()[:8]
    return safe or "page"


class _LinkExtractor(HTMLParser):
    """Extract same-domain absolute hrefs from an HTML page."""

    def __init__(self, base: str) -> None:
        super().__init__()
        self.base = base
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "a":
            return
        for name, val in attrs:
            if name == "href" and val:
                # Strip fragment, then resolve relative → absolute.
                abs_url = urllib.parse.urljoin(self.base, val.split("#")[0])
                parsed = urllib.parse.urlparse(abs_url)
                base_p = urllib.parse.urlparse(self.base)
                if (parsed.scheme in ("http", "https")
                        and parsed.netloc == base_p.netloc
                        and abs_url not in self.links):
                    self.links.append(abs_url)


def _fetch_page(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct and "xml" not in ct:
                return None
            raw = resp.read(_MAX_BYTES)
            return raw.decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        log.warning("fetch %s: %s", url, exc)
        return None


def fetch_link(
    link: ExternalLink,
    external_dir: Path,
    force: bool = False,
    cancel_check: Callable[[], None] | None = None,
    progress_cb: Callable[[int, int, int], None] | None = None,
) -> int:
    """Fetch `link` into `external_dir`. Returns the number of pages saved.

    Skips entirely when the link is still fresh and `force` is False.
    Pages are written as `<url_slug>.html` (seed) and `<url_slug>__p2.html`,
    `__p3.html` … for subsequent crawl pages.

    `cancel_check` is called between page fetches; it should raise to abort.
    `progress_cb(depth, done_at_depth, total_at_depth)` is called after each
    page save. `total_at_depth` grows as new links are discovered.
    `link.max_pages > 0` caps the BFS regardless of depth.

    Mutates `link.last_fetched` and `link.page_count` in place; the
    caller is responsible for calling `save_links()` afterward.
    """
    if not force and not link.is_stale():
        remaining = link.ttl_hours - (time.time() - (link.last_fetched or 0)) / 3600
        log.debug("fetch %s: fresh (%.1fh TTL remaining)", link.url, remaining)
        return link.page_count or 0

    external_dir.mkdir(parents=True, exist_ok=True)
    slug = _url_slug(link.url)
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(link.url, 0)]
    saved = 0
    max_pages = link.max_pages or 0
    # BFS progress tracking: how many URLs exist and are done per depth level.
    total_per_depth: dict[int, int] = {0: 1}
    done_per_depth: dict[int, int] = {}

    while queue:
        if cancel_check is not None:
            cancel_check()
        if max_pages and saved >= max_pages:
            log.debug("fetch %s: max_pages=%d reached, stopping", link.url, max_pages)
            break
        url, depth = queue.pop(0)
        if url in visited:
            continue
        if depth > link.depth:
            continue
        visited.add(url)

        html = _fetch_page(url)
        if html is None:
            continue

        suffix = "" if saved == 0 else f"__p{saved + 1}"
        out_path = external_dir / f"{slug}{suffix}.html"
        out_path.write_text(html, encoding="utf-8")
        saved += 1
        done_per_depth[depth] = done_per_depth.get(depth, 0) + 1
        log.info("fetch: %s → %s", url, out_path.name)

        if depth < link.depth:
            extractor = _LinkExtractor(url)
            try:
                extractor.feed(html)
            except Exception:
                pass
            new_links = [c for c in extractor.links if c not in visited]
            if new_links:
                total_per_depth[depth + 1] = (
                    total_per_depth.get(depth + 1, 0) + len(new_links)
                )
            for child in new_links:
                queue.append((child, depth + 1))

        if progress_cb is not None:
            try:
                progress_cb(
                    depth,
                    done_per_depth[depth],
                    total_per_depth.get(depth, 1),
                )
            except Exception:
                pass

    link.last_fetched = time.time()
    link.page_count = saved
    return saved


def fetch_all(
    data_dir: Path,
    force: bool = False,
    only_url: str | None = None,
    cancel_check: Callable[[], None] | None = None,
    progress_cb: Callable[[int, int, int], None] | None = None,
) -> dict[str, int]:
    """Fetch all (or one) external links registered for this root.

    Returns {url: pages_saved}. Stale links are re-fetched; fresh ones
    are skipped unless force=True. Writes updated timestamps back to
    links.json so subsequent runs respect the TTL.
    """
    links = load_links(data_dir)
    if not links:
        return {}

    external_dir = data_dir / "external"
    results: dict[str, int] = {}
    for lk in links:
        if only_url and lk.url != only_url:
            continue
        count = fetch_link(lk, external_dir, force=force,
                           cancel_check=cancel_check, progress_cb=progress_cb)
        results[lk.url] = count

    save_links(data_dir, links)
    return results
