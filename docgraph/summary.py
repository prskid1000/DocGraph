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
CHUNK_MAX_CHARS = 1400        # hard cap; if no scope boundary fits in target..max, fall back

# Per-language regex matching the start of a "scope" line — a method, function,
# nested class, or top-level visibility-keyword declaration. We split on these
# so a 2000-line class doesn't get sliced mid-method. Matches are anchored to
# the start of a stripped line so nested code (which is more deeply indented)
# is naturally avoided in most cases. Conservative: false negatives are fine
# (we fall back to line-based chunking); false positives split too aggressively
# but each chunk is still well-formed text.
_SCOPE_BOUNDARY_PATTERNS: dict[str, str] = {
    "python":     r"^(?:async\s+def|def|class|@)\s",
    "javascript": r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s|^[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{|^(?:get|set|static|async|#)\s+[A-Za-z_$]",
    "typescript": r"^(?:export\s+)?(?:abstract\s+|async\s+)?(?:function|class|interface|enum|namespace|type)\s|^(?:public|private|protected|readonly|static|abstract|async|get|set|#)\s+[A-Za-z_$]",
    "tsx":        r"^(?:export\s+)?(?:abstract\s+|async\s+)?(?:function|class|interface|enum|namespace|type)\s|^(?:public|private|protected|readonly|static|abstract|async|get|set|#)\s+[A-Za-z_$]",
    "java":       r"^(?:public|private|protected|static|final|abstract|synchronized|@)\s|^(?:class|interface|enum|record)\s",
    "c_sharp":    r"^(?:public|private|protected|internal|static|virtual|override|sealed|async|partial|\[)\s|^(?:class|interface|struct|enum|record|namespace)\s",
    "go":         r"^func\s|^type\s|^var\s|^const\s",
    "rust":       r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|mod|type|const|static)\s|^#\[",
    "c":          r"^(?:static\s+|inline\s+|extern\s+)*[A-Za-z_][\w*\s]*\([^)]*\)\s*\{?$",
    "cpp":        r"^(?:public|private|protected)\s*:|^(?:static\s+|inline\s+|virtual\s+|explicit\s+|template\s*<)|^(?:class|struct|enum|namespace)\s",
    "kotlin":     r"^(?:public|private|protected|internal|open|final|abstract|override|suspend|@)\s|^(?:fun|class|interface|object|enum)\s",
    "scala":      r"^(?:def|val|var|class|object|trait|case\s+class|implicit|@)\s",
    "ruby":       r"^(?:def|class|module|attr_)\s",
    "php":        r"^(?:public|private|protected|static|abstract|final)\s+function|^(?:class|interface|trait|namespace|function)\s",
    "elixir":     r"^(?:def|defp|defmodule|defmacro|defstruct|@)\s",
    "swift":      r"^(?:public|private|fileprivate|internal|open|static|class|struct|enum|protocol|extension|func|init)\s",
    "markdown":   r"^#{1,6}\s",
}


def _scope_boundary_lines(body: str, language: str | None) -> set[int]:
    """Return the set of line indices that start a new top-level scope.
    Empty when language is unknown or has no pattern. Indices match
    `body.splitlines()` ordering."""
    if not language:
        return set()
    pat = _SCOPE_BOUNDARY_PATTERNS.get(language)
    if not pat:
        return set()
    rx = re.compile(pat)
    out: set[int] = set()
    for i, ln in enumerate(body.splitlines()):
        stripped = ln.lstrip()
        if not stripped:
            continue
        if rx.match(stripped):
            out.add(i)
    return out


def chunk_body(body: str, language: str | None = None) -> list[str]:
    """Split a long body into chunks. Prefers scope boundaries (method /
    function / class declarations) when a `language` hint is provided so a
    long class isn't sliced mid-method. Falls back to a pure line+overlap
    splitter when no boundary is reachable in the target window. Returns []
    if the body is short enough that one embedding is sufficient."""
    if len(body) < CHUNK_MIN_BODY_CHARS:
        return []
    lines = body.splitlines(keepends=True)
    boundaries = _scope_boundary_lines(body, language)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    last_emit = -1  # index of the line where the previous chunk ended

    def flush(carry_overlap: bool) -> None:
        """Emit current buffer as a chunk. If carry_overlap, seed `cur` with
        the tail of the just-emitted chunk so cross-chunk context is
        preserved (used for hard-cap mid-method splits). For scope-aware
        splits we pass carry_overlap=False so the next method starts fresh
        and isn't preceded by garbage from the prior method's body."""
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append("".join(cur))
        if not carry_overlap:
            cur = []
            cur_len = 0
            return
        overlap_lines: list[str] = []
        overlap_len = 0
        for back in reversed(cur):
            if overlap_len >= CHUNK_OVERLAP_CHARS:
                break
            overlap_lines.insert(0, back)
            overlap_len += len(back)
        cur = list(overlap_lines)
        cur_len = overlap_len

    for i, ln in enumerate(lines):
        # Scope-aware split: if we're already over the target window AND the
        # *next* line starts a new scope, cut here. This keeps each method
        # whole instead of slicing across its body. No overlap on the cut —
        # the boundary is its own context.
        if (
            boundaries
            and cur_len >= CHUNK_TARGET_CHARS
            and i in boundaries
            and i > last_emit
        ):
            flush(carry_overlap=False)
            last_emit = i

        cur.append(ln)
        cur_len += len(ln)

        # Hard cap: even without a boundary, never grow past CHUNK_MAX_CHARS.
        # Below this we keep accumulating in the hope of hitting a boundary.
        # Carry overlap so a mid-method cut still has cross-chunk context.
        if cur_len >= CHUNK_MAX_CHARS:
            flush(carry_overlap=True)
            last_emit = i

    if cur:
        tail = "".join(cur)
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
    *,
    llm_doc: str | None = None,
) -> str:
    """Compose the text fed to the embedding model.

    Order matters — embedding models weight earlier tokens more, so we lead
    with the human-readable signal (name + signature + docstring) before
    falling back to raw code. `llm_doc` is the optional LLM-generated
    summary used when no native docstring exists.
    """
    docstring = extract_docstring(body, language)
    if not docstring and llm_doc:
        docstring = llm_doc.strip()
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
