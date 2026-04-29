"""Build the text we embed for each entity.

Concatenating raw `body[:1500]` is wasteful — embedding models lose signal in
verbose code. We extract leading docstrings / doc-comments and prepend the
signature so semantic queries match intent, not boilerplate.

This is the cheapest meaningful upgrade over Cursor / Greptile parity:
docstrings get higher weight and we keep the model on-device.
"""
from __future__ import annotations

import re

# Heuristic max-chars budget for the embedding text. BGE-small handles ~512
# tokens; ~2200 chars is a comfortable upper bound that leaves headroom.
MAX_EMBED_CHARS = 2200

# How much of the raw body to keep when no docstring is found.
RAW_BODY_HEAD = 900
RAW_BODY_TAIL = 400


def extract_docstring(body: str, language: str) -> str:
    """Pull the leading docstring or doc-comment out of an entity body.

    Returns "" if no doc-block is present. Conservative: false-negatives are
    fine, but false-positives would inject random code into the embedding.
    """
    if not body:
        return ""

    # Python: triple-quoted string as the first statement of the body.
    if language in ("python",):
        m = re.search(
            r'^\s*(?:def |class |async def )[^\n]*\n\s*([rRbBuU]?(?:"""|\'\'\')(.*?)(?:"""|\'\'\'))',
            body,
            re.DOTALL,
        )
        if m:
            return _clean(m.group(2))
        # Some indexers feed us just the body without the def line.
        m = re.match(r'^\s*([rRbBuU]?(?:"""|\'\'\')(.*?)(?:"""|\'\'\'))', body, re.DOTALL)
        if m:
            return _clean(m.group(2))

    # JS/TS/Java/C/C++/C#/Go/Rust: /** ... */ JSDoc block immediately above
    # or as the first non-blank thing in the body.
    if language in (
        "javascript", "typescript", "tsx", "java", "go", "rust",
        "c", "cpp", "c_sharp", "php",
    ):
        m = re.search(r"/\*\*?(.*?)\*/", body, re.DOTALL)
        if m and m.start() < 200:
            return _clean(re.sub(r"\n\s*\*\s?", "\n", m.group(1)))
        # Consecutive `///` (Rust/C#) or `//` doc lines at the very top.
        head_lines = body.splitlines()[:8]
        doc_lines: list[str] = []
        for ln in head_lines:
            s = ln.strip()
            if s.startswith("///"):
                doc_lines.append(s.lstrip("/").strip())
            elif s.startswith("//") and doc_lines:
                doc_lines.append(s.lstrip("/").strip())
            elif s and not s.startswith("//") and doc_lines:
                break
        if doc_lines:
            return _clean(" ".join(doc_lines))

    # Ruby: =begin ... =end or leading `# ...` lines.
    if language == "ruby":
        m = re.search(r"=begin(.*?)=end", body, re.DOTALL)
        if m:
            return _clean(m.group(1))
        head = []
        for ln in body.splitlines()[:6]:
            s = ln.strip()
            if s.startswith("#"):
                head.append(s.lstrip("#").strip())
            elif s and head:
                break
        if head:
            return _clean(" ".join(head))

    return ""


def _clean(text: str) -> str:
    text = text.strip()
    # Collapse whitespace; keep paragraph breaks lightly preserved.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:1200]


def smart_body_sample(body: str) -> str:
    """Take head + tail of a long body so we cover signature + return logic."""
    if len(body) <= RAW_BODY_HEAD + RAW_BODY_TAIL:
        return body
    return body[:RAW_BODY_HEAD] + "\n…\n" + body[-RAW_BODY_TAIL:]


CHUNK_MIN_BODY_CHARS = 1500   # below this, no sub-chunks created
CHUNK_TARGET_CHARS = 700      # target size of each chunk
CHUNK_OVERLAP_CHARS = 80      # overlap between adjacent chunks (preserves cross-chunk context)


def chunk_body(body: str) -> list[str]:
    """Split a long body into ~CHUNK_TARGET_CHARS chunks aligned to line
    boundaries. Returns [] if the body is short enough that one embedding
    is sufficient."""
    if len(body) < CHUNK_MIN_BODY_CHARS:
        return []
    lines = body.splitlines(keepends=True)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lines:
        cur.append(ln)
        cur_len += len(ln)
        if cur_len >= CHUNK_TARGET_CHARS:
            chunks.append("".join(cur))
            # Overlap: rewind a bit so the next chunk starts in mid-context.
            overlap_lines: list[str] = []
            overlap_len = 0
            for back in reversed(cur):
                if overlap_len >= CHUNK_OVERLAP_CHARS:
                    break
                overlap_lines.insert(0, back)
                overlap_len += len(back)
            cur = list(overlap_lines)
            cur_len = overlap_len
    if cur and (not chunks or "".join(cur) != chunks[-1]):
        tail = "".join(cur)
        # Skip if tail is just the overlap of the last chunk (i.e. no new content)
        if not chunks or tail.strip() != chunks[-1][-len(tail):].strip():
            chunks.append(tail)
    return [c for c in chunks if c.strip()]


def build_embedding_text(
    name: str,
    qname: str,
    signature: str,
    body: str,
    language: str,
    kind: str,
) -> str:
    """Compose the text fed to the embedding model.

    Order matters — embedding models weight earlier tokens more, so we lead
    with the human-readable signal (name + signature + docstring) before
    falling back to raw code.
    """
    docstring = extract_docstring(body, language)
    parts: list[str] = []
    parts.append(f"{kind} {name}")
    if signature and signature != name:
        parts.append(signature.strip())
    if qname and qname != name:
        parts.append(qname)
    if docstring:
        parts.append(docstring)
    parts.append(smart_body_sample(body))
    text = "\n".join(p for p in parts if p)
    return text[:MAX_EMBED_CHARS]
