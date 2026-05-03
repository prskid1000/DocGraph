"""Document + asset indexing for repo files outside the code tier.

Three tiers, all driven from the same walker pass:

- **Code** (existing) — tree-sitter parsed, full graph.
- **Light text docs** — `.md` / `.markdown` / `.txt` / `.rst` and small
  CSVs. Content is extracted (stdlib only — no pypdf / openpyxl /
  python-docx), chunked, embedded, and stored as `Doc` nodes whose
  `source` is the logical relative path.
- **Media / large / binary** — `.pdf` / `.xlsx` / `.docx` / `.png` /
  `.mp4` / `.parquet` / fonts / archives / 3D assets. We DO NOT
  extract their content; we register an `Asset` node (path + size +
  ext + mime) and emit `REFERENCES_` edges from any code or doc file
  whose text contains a path-like reference to that asset.

Reference extraction is **format-aware**, not raw regex:

- Markdown: `![]()` / `[]()` link syntax via a deliberate parse, plus
  inline backticks and HTML `<img src>` / `<a href>` inside markdown.
- HTML: `src=""` / `href=""` / `data-src=""` attributes.
- Code: only **quoted string literals** with a known asset extension
  are considered (avoids matching identifiers, comments, SQL
  fragments, etc.). The quote-anchored regex is much stricter than
  matching free-floating words.
- Plain text / RST: anchored extension regex over the full body.

Path normalization:
- Strip a leading `./`, `/`, `~/`, scheme prefixes (`file://`).
- Replace `\\` with `/` so Windows-authored repos match.
- Lowercase the file extension (so `.PDF` matches `.pdf`).
- Strip `?query=` and `#fragment` so `data.csv?cache=1` matches `data.csv`.

CSV is borderline by design — small CSVs are useful as text (column
names + sample rows search well), large ones become noise when
embedded. We size-gate at 1 MiB by default; CSVs larger than that
fall through to the asset tier instead.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
from html.parser import HTMLParser
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


# Sensible defaults — overridable via CLI flag or env var (DOCGRAPH_TEXT_EXTS
# / DOCGRAPH_ASSET_EXTS, comma-separated, no leading dots).
DEFAULT_TEXT_EXTS: tuple[str, ...] = ("md", "markdown", "txt", "rst", "csv")
DEFAULT_ASSET_EXTS: tuple[str, ...] = (
    "pdf", "xlsx", "xls", "docx", "doc", "ppt", "pptx",
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "bmp", "tiff",
    "mp4", "mov", "webm", "avi", "mkv", "mp3", "wav", "flac", "ogg", "m4a",
    "zip", "tar", "gz", "tgz", "7z", "rar", "bz2", "xz",
    "parquet", "feather", "arrow", "h5", "hdf5", "pkl", "pickle", "npz", "npy",
    "ttf", "woff", "woff2", "otf", "eot",
    "gltf", "glb", "fbx", "obj", "stl", "blend",
    "psd", "ai", "indd", "fig",
    "iso", "dmg", "exe", "deb", "rpm", "msi",
)
DEFAULT_CSV_TEXT_MAX_BYTES = 1_048_576  # 1 MiB
DEFAULT_DOC_FILE_MAX_BYTES = 5 * 1_048_576  # 5 MiB cap on text doc extraction


# ── Path normalization ─────────────────────────────────────────────────

_SCHEME_PREFIX_RE = re.compile(r"^(?:file://|\./|/|~/)+")


def normalize_path(s: str) -> str:
    """Lossily normalize a candidate path string for asset matching.

    Doesn't try to be filesystem-safe — just gets two strings into the
    same shape so `references("docs/img.png")` matches `./docs/img.png`
    in the source code. Stripping the leading slash means absolute
    repo paths and relative paths both collapse onto the same shape;
    that's intentional.
    """
    if not s:
        return ""
    out = s.strip()
    out = out.replace("\\", "/")
    out = _SCHEME_PREFIX_RE.sub("", out)
    # Strip query / fragment
    for sep in ("?", "#"):
        i = out.find(sep)
        if i >= 0:
            out = out[:i]
    # Lowercase the extension only — preserve filename case for case-
    # sensitive filesystems. We split on the last `.`.
    dot = out.rfind(".")
    if dot >= 0:
        out = out[:dot] + out[dot:].lower()
    return out


# ── Text extractors (stdlib only) ──────────────────────────────────────

_RST_INLINE_RE = re.compile(r":[a-z]+:`([^`]+)`")  # strip RST roles
_MD_HTML_RE = re.compile(r"<[^>]+>")  # strip raw HTML for plain reading


def extract_text(path: Path, ext: str) -> tuple[str, str]:
    """Read a text doc and return (title, body). Falls back to filename
    as title when no leading heading is present. Empty (title, "") on
    unreadable / encoding-failure."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.debug("documents.extract_text(%s): %s", path, exc)
        return path.stem, ""
    ext = (ext or "").lower()
    if ext in ("md", "markdown"):
        title = _markdown_title(raw) or path.stem
        return title, raw
    if ext == "rst":
        title = _rst_title(raw) or path.stem
        # Strip role inline markers for cleaner embedding text
        body = _RST_INLINE_RE.sub(r"\1", raw)
        return title, body
    if ext == "csv":
        return path.stem, _csv_body(raw)
    return path.stem, raw  # txt / unknown


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _rst_title(text: str) -> str | None:
    """RST titles are 'Title\\n=====' — find the first line whose next
    line is composed entirely of one underline character."""
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if not title or len(underline) < 3:
            continue
        if len(set(underline)) == 1 and underline[0] in "=-~^*+#`'\"":
            return title
    return None


def _csv_body(raw: str) -> str:
    """Render a CSV as a search-friendly text blob: header row, a few
    sample rows, then column-wise summaries. Cheap, no pandas dep.

    For very wide tables we cap the per-row width so embeddings don't
    drown in commas."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = [f"CSV header: {lines[0][:1000]}"]
    sample = lines[1:6]
    if sample:
        out.append("Sample rows:")
        out.extend(f"  {ln[:1000]}" for ln in sample)
    if len(lines) > 6:
        out.append(f"(+{len(lines) - 6} more rows)")
    return "\n".join(out)


# ── Walking the repo ───────────────────────────────────────────────────

def _ext_of(name: str) -> str:
    dot = name.rfind(".")
    return name[dot + 1:].lower() if dot >= 0 else ""


def walk_documents(
    cfg: Config,
    text_exts: tuple[str, ...] = DEFAULT_TEXT_EXTS,
    asset_exts: tuple[str, ...] = DEFAULT_ASSET_EXTS,
    csv_text_max_bytes: int = DEFAULT_CSV_TEXT_MAX_BYTES,
    doc_file_max_bytes: int = DEFAULT_DOC_FILE_MAX_BYTES,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str, int]]]:
    """Walk every registered repo root and bucket files into:

      - text_docs:  list of (absolute_path, logical_rel, ext)
      - assets:     list of (absolute_path, logical_rel, ext, size_bytes)

    Respects every Config ignore tier (.gitignore / .docgraphignore /
    ecosystem auto-detect) — uses the same mechanism the code walker
    does so the two passes never disagree about what "in the repo" means.

    CSVs above `csv_text_max_bytes` get reclassified into the asset
    tier — embedding a 50 MB analytics dump is noise, but `references`
    edges to it are still useful.
    """
    text_set = {e.lower().lstrip(".") for e in text_exts}
    asset_set = {e.lower().lstrip(".") for e in asset_exts}
    text_docs: list[tuple[Path, str, str]] = []
    assets: list[tuple[Path, str, str, int]] = []

    for root, prefix in cfg.roots_with_prefix():
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
            # Directory-level filtering uses the FULL ignore spec — we
            # never want to descend into node_modules / .git / venv /
            # etc. even when looking for documents. File-level filtering
            # uses the user-only spec so the universal `*.pdf` / `*.png`
            # exclusions don't pre-empt the document tier.
            dirnames[:] = [
                d for d in dirnames
                if not cfg.is_ignored(
                    f"{rel_dir}/{d}/" if rel_dir != "." else f"{d}/", root=root
                )
            ]
            for fname in filenames:
                ext = _ext_of(fname)
                if ext not in text_set and ext not in asset_set:
                    continue
                full = Path(dirpath) / fname
                rel = str(full.relative_to(root)).replace("\\", "/")
                if cfg.is_user_ignored(rel, root=root):
                    continue
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                logical = f"{prefix}{rel}"
                if ext in text_set:
                    if ext == "csv" and size > csv_text_max_bytes:
                        if ext in asset_set or "csv" in asset_set:
                            assets.append((full, logical, "csv", size))
                        continue
                    if size > doc_file_max_bytes:
                        # too big to embed as text — drop into assets if
                        # the user has the extension in their asset list,
                        # otherwise skip silently.
                        if ext in asset_set:
                            assets.append((full, logical, ext, size))
                        continue
                    text_docs.append((full, logical, ext))
                elif ext in asset_set:
                    assets.append((full, logical, ext, size))
    return text_docs, assets


# ── Reference extraction (format-aware) ────────────────────────────────

# Quoted string literal containing a path-like value. The path must have
# at least one extension and may contain forward or back slashes.
_QUOTED_PATH_RE = re.compile(r'''(?P<q>['"`])([^'"`\n\r]{2,512}\.([A-Za-z0-9]{1,6}))(?P=q)''')

# Markdown link / image: `![alt](path "title")` or `[text](path "title")`.
_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s\"'<>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
# Markdown reference-style: `[label]: path "title"`
_MD_REF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*<?([^\s>]+)>?", re.MULTILINE)

# Anchored extension probe — used as the final filter on extracted candidates.
_EXT_RE = re.compile(r"\.([A-Za-z0-9]{1,6})$")


class _SrcHrefCollector(HTMLParser):
    """Pull every src/href/data-src/poster-style attribute out of HTML or
    embedded HTML inside Markdown."""
    _ATTR_KEYS = (
        "src", "href", "data-src", "data-href", "poster",
        "data-asset", "background",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for k, v in attrs:
            if v and k.lower() in self._ATTR_KEYS:
                self.found.append(v)


def extract_references_markdown(text: str) -> set[str]:
    """Pull every path-shaped reference out of a markdown document."""
    out: set[str] = set()
    for m in _MD_LINK_RE.finditer(text):
        out.add(m.group(1))
    for m in _MD_REF_RE.finditer(text):
        out.add(m.group(1))
    # Markdown commonly contains literal HTML for embeds.
    p = _SrcHrefCollector()
    try:
        p.feed(text)
    except Exception:
        pass
    out.update(p.found)
    # Inline code paths — backtick-wrapped strings ending in a known ext.
    for m in re.finditer(r"`([^`\n]{2,256})`", text):
        cand = m.group(1)
        if _EXT_RE.search(cand):
            out.add(cand)
    return out


def extract_references_html(text: str) -> set[str]:
    p = _SrcHrefCollector()
    try:
        p.feed(text)
    except Exception:
        pass
    return set(p.found)


def extract_references_code(text: str) -> set[str]:
    """Pull path-like quoted string literals out of source code.

    Improvement over a free-floating regex: we ONLY accept matches that
    appear inside matching quote characters (`"…"`, `'…'`, or `` `…` ``)
    AND end with a 1-6 char extension. This skips identifiers,
    comments-as-text, SQL fragments, and most accidental matches.

    We do NOT try to resolve dynamic concatenation (`f"docs/{name}.pdf"`)
    — the design promises explicit references only. A path that starts
    with a static prefix and ends with an extension via concatenation
    is still picked up if BOTH halves are visible in one literal,
    because the regex matches the whole quoted span.
    """
    out: set[str] = set()
    for m in _QUOTED_PATH_RE.finditer(text):
        cand = m.group(2)
        # Skip false positives: floating-point literals (e.g. "3.14"),
        # version strings ("1.2.3"), single-char "extensions" that are
        # actually digits.
        ext = m.group(3).lower()
        if ext.isdigit():
            continue
        # Skip strings that are obviously not paths (no separators AND
        # no recognizable extension shape). One-word filenames like
        # "logo.png" are kept because the extension is what we match on.
        if "/" not in cand and "\\" not in cand:
            # Single-segment filename — keep only if the extension
            # is at least 2 chars (skips noise like "x.5").
            if len(ext) < 2:
                continue
        out.add(cand)
    return out


def extract_references_text(text: str) -> set[str]:
    """Last-resort scanner for plain `.txt` / `.rst` files. Pulls
    whitespace-bounded tokens that end in a known-looking extension."""
    out: set[str] = set()
    for m in re.finditer(r"(?:^|[\s\(\[])([\w./\\-]{2,256}\.[A-Za-z0-9]{1,6})(?=[\s\)\],;:!?]|$)", text):
        out.add(m.group(1))
    return out


def extract_references_for(content: str, ext: str) -> set[str]:
    """Dispatch by extension to the right format-aware extractor.
    Returns a set of raw (un-normalized) candidate paths."""
    e = (ext or "").lower()
    if e in ("md", "markdown", "rst"):
        return extract_references_markdown(content) | extract_references_text(content)
    if e in ("html", "htm"):
        return extract_references_html(content)
    if e in ("txt",):
        return extract_references_text(content)
    if e == "csv":
        # CSVs commonly contain paths in cells. Plain text scan is fine.
        return extract_references_text(content)
    # Default to the strict quoted-string scanner for code files.
    return extract_references_code(content)


# ── Asset matching ─────────────────────────────────────────────────────

def build_asset_lookup(assets: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Build a `normalized_path -> [asset_id, ...]` index.

    For each asset we register both the full path and just the basename
    so a `references("logo.png")` style mention resolves even when the
    code has `<img src="logo.png">` without the directory. This trades
    some precision (collisions on basename) for recall (mentions
    without explicit paths).
    """
    by_key: dict[str, list[int]] = {}
    for path, asset_id in assets:
        norm = normalize_path(path)
        if not norm:
            continue
        by_key.setdefault(norm, []).append(asset_id)
        # Also key by basename for the bare-filename case
        slash = norm.rfind("/")
        base = norm[slash + 1:] if slash >= 0 else norm
        if base != norm:
            by_key.setdefault(base, []).append(asset_id)
    return by_key


def resolve_to_assets(candidates: set[str], lookup: dict[str, list[int]]) -> set[int]:
    """Map a set of raw candidate path strings to Asset ids.

    A candidate matches if its normalized form OR its trailing path
    segment lookup-hits. We dedupe by id but keep all collisions."""
    hits: set[int] = set()
    for raw in candidates:
        n = normalize_path(raw)
        if not n:
            continue
        if n in lookup:
            hits.update(lookup[n])
            continue
        # Try suffix matches: candidate ends with an asset's normalized
        # path — covers absolute references in HTML when the repo path
        # is relative.
        slash = n.rfind("/")
        base = n[slash + 1:] if slash >= 0 else n
        if base in lookup and base != n:
            hits.update(lookup[base])
    return hits


def mime_for(ext: str) -> str:
    """Best-effort MIME type from extension. mimetypes.guess_type wants
    a filename so we fake one. Falls back to `application/octet-stream`."""
    t, _ = mimetypes.guess_type(f"file.{ext}")
    return t or "application/octet-stream"


def is_url_source(source: str) -> bool:
    """Distinguish `Doc.source_kind`: URL-fetched (`@Docs`) vs file-tier."""
    if not source:
        return False
    s = source.lower()
    return s.startswith("http://") or s.startswith("https://")
