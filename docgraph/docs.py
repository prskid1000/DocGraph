"""External documentation ingestion (Cursor `@Docs` parity).

Fetches a URL, strips HTML to text, chunks, embeds, stores as Doc nodes.
Uses stdlib HTML parser to avoid pulling in BeautifulSoup as a dep.

Default user agent identifies docgraph so docs sites don't return a
generic crawler-block page.
"""
from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

from rich.console import Console

from docgraph.cancel import CancelToken
from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder, GPU_PROVIDERS, resolve_providers
from docgraph.index import _bar
from docgraph.summary import chunk_body

_console = Console()

log = logging.getLogger(__name__)

USER_AGENT = "docgraph/2.0 (+https://github.com/prskid1000/DocGraph)"
TIMEOUT_SECS = 20
MAX_BYTES = 5_000_000

DOC_CHUNK_TARGET_CHARS = 800


# --- HTML → text ------------------------------------------------------


_SKIP_TAGS = {"script", "style", "noscript", "head", "header", "footer", "nav"}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"):
            self._parts.append("\n")
        elif tag == "pre":
            self._parts.append("\n```\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "pre":
            self._parts.append("\n```\n")

    def handle_data(self, data: str):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace; preserve blank lines as separators
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, plain_text)."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception as e:  # noqa: BLE001
        log.debug(f"html parse error: {e}")
    return parser.title.strip(), parser.text()


# --- Fetch + chunk ----------------------------------------------------


def fetch_url(url: str) -> tuple[str, str]:
    """GET `url`, return (title, body_text). Raises URLError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(MAX_BYTES)
    text = raw.decode("utf-8", errors="replace")
    if "html" in ctype.lower():
        return html_to_text(text)
    # Treat anything else (markdown, plain text) as already plain
    title = ""
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if first_line.startswith("#"):
        title = first_line.lstrip("#").strip()
    return title, text


def chunk_doc(text: str) -> list[str]:
    """Doc chunking is a bit more aggressive than entity chunking — pages are
    long and we want denser retrieval. Reuses the line-aligned splitter but
    with a smaller target."""
    if len(text) <= DOC_CHUNK_TARGET_CHARS:
        return [text]
    # Build chunks of ~DOC_CHUNK_TARGET_CHARS preserving paragraph breaks
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for para in paragraphs:
        if cur_len + len(para) + 2 > DOC_CHUNK_TARGET_CHARS and cur:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(para)
        cur_len += len(para) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]


# --- Public API -------------------------------------------------------


def add_doc(
    cfg: Config,
    url: str,
    db: GraphDB | None = None,
    cancel_token: CancelToken | None = None,
) -> dict:
    """Fetch `url`, chunk, embed, write Doc rows. Replaces any existing Doc
    rows for the same source URL (so re-running is idempotent).

    Pass an existing writer `db` to reuse it (the host's `/api/docs/add`
    route does this so it doesn't fight the workspace's writer-lock dance
    by opening a parallel GraphDB).

    Pass `cancel_token` to make the op cooperatively cancellable. Checkpoints
    are at safe boundaries (before fetch, after fetch, before embed, before
    insert) — never mid-embed (ONNX batch) or mid-COPY (Kuzu)."""
    def _ck() -> None:
        if cancel_token is not None:
            cancel_token.raise_if_set()

    _ck()
    with _console.status(f"[cyan]Fetching[/] {url}"):
        title, text = fetch_url(url)
    _ck()
    if not text.strip():
        return {"url": url, "chunks": 0, "title": title, "error": "empty body"}

    if db is None:
        db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()

    # Remove existing chunks for this URL
    try:
        db.execute("MATCH (d:Doc) WHERE d.source = $u DETACH DELETE d", {"u": url})
    except Exception:
        pass

    pieces = chunk_doc(text)
    _ck()
    embedder = Embedder(
        cfg.embedding_model,
        providers=resolve_providers(cfg.gpu, getattr(cfg, "directml_device_id", -1)),
    )
    with _bar() as prog:
        task = prog.add_task(f"Embedding doc chunks ({title or url})", total=len(pieces))
        vecs = embedder.embed(
            pieces,
            batch_size=cfg.embed_batch_size,
            on_progress=lambda n: prog.advance(task, n),
        )
    _ck()

    # Continue id allocation past max(id) across all node tables
    max_id = 0
    for label in ("File", "Module", "Class", "Function", "Variable", "Chunk", "Doc"):
        try:
            rows = db.fetch_all(f"MATCH (n:{label}) RETURN max(n.id) AS m")
            m = rows[0]["m"] if rows and rows[0]["m"] is not None else 0
            if m > max_id:
                max_id = m
        except Exception:
            pass

    rows = []
    for i, (piece, vec) in enumerate(zip(pieces, vecs)):
        max_id += 1
        rows.append({
            "id": max_id,
            "source": url,
            "title": title or urlparse(url).netloc,
            "idx": i,
            "body": piece[:6000],
            "embedding": vec,
        })
    _ck()
    db.insert_nodes("Doc", rows)
    return {"url": url, "title": title, "chunks": len(rows)}


def list_docs(cfg: Config) -> list[dict]:
    db = GraphDB(cfg.db_path, read_only=True)
    try:
        rows = db.fetch_all(
            "MATCH (d:Doc) RETURN d.source AS source, d.title AS title, count(d) AS chunks "
            "GROUP BY d.source, d.title ORDER BY d.source"
        )
    except Exception:
        # Older Kuzu doesn't support GROUP BY in this form; fall back
        all_rows = db.fetch_all("MATCH (d:Doc) RETURN d.source AS source, d.title AS title")
        bucket: dict[tuple[str, str], int] = {}
        for r in all_rows:
            key = (r["source"], r["title"])
            bucket[key] = bucket.get(key, 0) + 1
        rows = [
            {"source": s, "title": t, "chunks": c}
            for (s, t), c in sorted(bucket.items())
        ]
    return rows


def remove_doc(cfg: Config, url: str, db: GraphDB | None = None) -> int:
    """Delete all Doc chunks for a given source URL. Returns count removed.

    Pass an existing writer `db` to reuse it (host route)."""
    if db is None:
        db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()
    rows = db.fetch_all(
        "MATCH (d:Doc) WHERE d.source = $u RETURN count(d) AS c", {"u": url}
    )
    n = rows[0]["c"] if rows else 0
    if n:
        db.execute("MATCH (d:Doc) WHERE d.source = $u DETACH DELETE d", {"u": url})
    return int(n or 0)
