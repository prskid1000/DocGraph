"""External-link registry per docgraph root.

Persisted at `<root>/.docgraph/links.json`. Each entry records the URL,
crawl depth, TTL, and last-fetched timestamp so the pre-index fetch step
can skip URLs that are still fresh.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExternalLink:
    url: str
    depth: int = 1
    ttl_hours: float = 24.0
    last_fetched: float | None = None   # Unix timestamp; None = never fetched
    page_count: int | None = None       # pages saved on last successful fetch

    def is_stale(self) -> bool:
        if self.last_fetched is None:
            return True
        return (time.time() - self.last_fetched) > self.ttl_hours * 3600


_FILENAME = "links.json"


def _path(data_dir: Path) -> Path:
    return data_dir / _FILENAME


def load_links(data_dir: Path) -> list[ExternalLink]:
    p = _path(data_dir)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: list[ExternalLink] = []
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("url"):
                continue
            out.append(ExternalLink(
                url=str(entry["url"]),
                depth=int(entry.get("depth", 1)),
                ttl_hours=float(entry.get("ttl_hours", 24.0)),
                last_fetched=float(entry["last_fetched"])
                    if entry.get("last_fetched") is not None else None,
                page_count=int(entry["page_count"])
                    if entry.get("page_count") is not None else None,
            ))
        return out
    except Exception:
        return []


def save_links(data_dir: Path, links: list[ExternalLink]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _path(data_dir).write_text(
        json.dumps([asdict(lk) for lk in links], indent=2),
        encoding="utf-8",
    )


def upsert_link(
    data_dir: Path, url: str, depth: int = 1, ttl_hours: float = 24.0
) -> list[ExternalLink]:
    """Add or update a link by URL. Returns the updated list."""
    links = load_links(data_dir)
    for lk in links:
        if lk.url == url:
            lk.depth = depth
            lk.ttl_hours = ttl_hours
            save_links(data_dir, links)
            return links
    links.append(ExternalLink(url=url, depth=depth, ttl_hours=ttl_hours))
    save_links(data_dir, links)
    return links


def remove_link(data_dir: Path, url: str) -> list[ExternalLink]:
    """Remove a link by URL. Returns the updated list."""
    links = [lk for lk in load_links(data_dir) if lk.url != url]
    save_links(data_dir, links)
    return links
