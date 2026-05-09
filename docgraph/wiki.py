"""LLM-generated wiki pages, grounded in the graph.

For each top-level module/package in the indexed repo we collect a small
fact sheet from Kuzu (top classes by PageRank, top functions, importers,
imported modules, sub-files), feed it to the LLM, and save the resulting
Markdown to `.docgraph/wiki/<slug>.md`. The web UI lists and renders these
pages; agents can also read them directly.

Off by default. Triggered by `docgraph wiki` or via `/api/wiki/build`.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.llm import LLMClient, LLMConfig

log = logging.getLogger(__name__)

WIKI_DIRNAME = "wiki"
INDEX_FILE = "index.json"


@dataclass
class WikiPage:
    slug: str
    title: str
    module: str
    summary: str  # short one-line description
    body_md: str  # full markdown
    facts: dict   # the fact sheet (so we can re-render without re-LLMing)


def _slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_/.-]+", "_", s).strip("_/")
    s = s.replace("/", "__").replace("\\", "__").replace(".", "_")
    return s.lower()[:120] or "module"


def _module_groupings(db: GraphDB, depth: int = 12) -> list[dict]:
    """Group all File rows by their containing directory, truncated to at
    most `depth` directory levels. Each file lands in exactly one bucket.

    `depth=1` = old behavior (top-level dir only). `depth=12` (default) is
    effectively "use each file's actual containing directory" for any
    repo shallower than 12 levels — i.e. one wiki page per leaf folder.

    Files at the repo root land in the "(root)" bucket.

    Ignored paths (`node_modules/`, `.venv/`, ecosystem build dirs, etc.)
    are inherited from index time — they were never inserted as File nodes,
    so they can't surface here.
    """
    rows = db.fetch_all("MATCH (f:File) RETURN f.path AS path")
    groups: dict[str, list[str]] = defaultdict(list)
    d = max(1, int(depth))
    for r in rows:
        path = (r.get("path") or "").replace("\\", "/")
        if not path:
            continue
        parts = path.split("/")
        if len(parts) <= 1:
            key = "(root)"
        else:
            # All parts except the filename, capped at `depth` segments.
            dirs = parts[:-1]
            key = "/".join(dirs[:d])
        groups[key].append(path)
    out = []
    for k in sorted(groups):
        files = sorted(groups[k])
        if not files:
            continue
        out.append({"module": k, "files": files})
    return out


def _facts_for_module(db: GraphDB, module: str, files: list[str]) -> dict:
    """Build a fact sheet for one module from the graph."""
    file_set = set(files)

    def _safe(query: str, params: dict | None = None) -> list[dict]:
        try:
            return db.fetch_all(query, params or {})
        except Exception as e:
            log.debug("wiki query failed: %s", e)
            return []

    # Top classes / functions in this module by PageRank. Use the AI-generated
    # llm_doc if available; otherwise fall back to the first line of the body.
    top_classes = _safe(
        "MATCH (c:Class) WHERE c.file IN $files "
        "RETURN c.name AS name, c.qname AS qname, c.file AS file, "
        "c.line_start AS line, coalesce(c.pagerank, 0.0) AS pagerank, "
        "c.body AS body, c.llm_doc AS llm_doc "
        "ORDER BY pagerank DESC LIMIT 8",
        {"files": files},
    )
    top_functions = _safe(
        "MATCH (f:Function) WHERE f.file IN $files "
        "RETURN f.name AS name, f.qname AS qname, f.file AS file, "
        "f.line_start AS line, coalesce(f.pagerank, 0.0) AS pagerank, "
        "f.body AS body, f.llm_doc AS llm_doc "
        "ORDER BY pagerank DESC LIMIT 12",
        {"files": files},
    )
    # Body → first interesting line as a doc-ish snippet
    for r in (*top_classes, *top_functions):
        if r.get("llm_doc"):
            r["doc"] = r["llm_doc"]
        else:
            b = (r.get("body") or "").strip().splitlines()
            snippet = next((ln.strip(' "\'') for ln in b if ln.strip() and not ln.strip().startswith(("def ", "class ", "@"))), "")
            r["doc"] = snippet[:160]
        r.pop("body", None)
        r.pop("llm_doc", None)
    # What this module imports (external modules / files outside it). Note: File
    # uses `path`, not `file`, as its locator property.
    imports = _safe(
        "MATCH (a:File)-[:IMPORTS]->(b) WHERE a.path IN $files "
        "RETURN coalesce(b.path, b.name) AS target, label(b) AS kind",
        {"files": files},
    )
    imports = [r for r in imports if r.get("target") not in file_set]
    # Who imports this module
    importers = _safe(
        "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE b.path IN $files AND NOT a.path IN $files "
        "RETURN DISTINCT a.path AS file LIMIT 20",
        {"files": files},
    )
    # Tests that exercise this module — `target` may be a File or a symbol.
    tests = _safe(
        "MATCH (t)-[:TESTS]->(target) WHERE coalesce(target.file, target.path) IN $files "
        "RETURN DISTINCT t.name AS name, t.file AS file LIMIT 10",
        {"files": files},
    )
    return {
        "module": module,
        "file_count": len(files),
        "files": files[:20],
        "top_classes": top_classes,
        "top_functions": top_functions,
        "imports": imports[:20],
        "importers": importers,
        "tests": tests,
    }


def _wiki_prompt(facts: dict) -> str:
    """Build a prompt from a fact sheet. Asks for grounded prose.

    Args:
        facts: The fact sheet for a module (classes, functions, imports, etc.)
    """
    parts: list[str] = [
        f"You are writing a documentation page for the `{facts['module']}` module of a codebase.",
        f"It contains {facts['file_count']} files. Use ONLY the facts below — do not invent.",
        "",
        "## Facts",
        f"- Module: {facts['module']}",
        f"- Files (sample): {', '.join(facts['files']) or '(none)'}",
    ]
    if facts.get("top_classes"):
        parts.append("- Top classes (by structural importance):")
        for c in facts["top_classes"][:6]:
            lines = (c.get("doc") or "").splitlines()
            doc = (lines[0][:120] if lines else "")
            parts.append(f"  - `{c.get('name','')}` ({c.get('file','')}:{c.get('line') or 0}) — {doc}")
    if facts.get("top_functions"):
        parts.append("- Top functions (by structural importance):")
        for f in facts["top_functions"][:8]:
            lines = (f.get("doc") or "").splitlines()
            doc = (lines[0][:120] if lines else "")
            parts.append(f"  - `{f.get('name','')}` ({f.get('file','')}:{f.get('line') or 0}) — {doc}")
    if facts.get("imports"):
        targets = ", ".join(sorted({i.get("target", "") for i in facts["imports"] if i.get("target")})[:10])
        parts.append(f"- This module imports: {targets}")
    if facts.get("importers"):
        callers = ", ".join(sorted({i.get("file", "") for i in facts["importers"]})[:8])
        parts.append(f"- Other parts of the codebase that import it: {callers}")
    if facts.get("tests"):
        tests = ", ".join(sorted({t.get("file", "") for t in facts["tests"]}))
        parts.append(f"- Tests covering it: {tests}")

    parts += ["", "## Output format", _wiki_output_format()]
    return "\n".join(parts)


_DEFAULT_WIKI_TAIL = (
    "Write a Markdown page with these sections:\n"
    "1. **Summary** — 2-3 sentences on the module's purpose.\n"
    "2. **Key entities** — bulleted list of the most important classes/functions and what each is for.\n"
    "3. **How it's used** — who imports it, in plain language.\n"
    "Total length: 200-300 words. No code blocks. Do not list every file. "
    "Only state what the facts support."
)


_WIKI_TAIL_OVERRIDE: str | None = None


def set_wiki_prompt_tail(text: str | None) -> None:
    """Install a process-wide override for the wiki output-format tail.
    Pass None / "" to revert to the built-in. The override does NOT need
    any placeholder substitution — the rendered facts sit above this
    section, and the override is appended verbatim."""
    global _WIKI_TAIL_OVERRIDE
    _WIKI_TAIL_OVERRIDE = text if (text and text.strip()) else None


def _wiki_output_format() -> str:
    """Tail of the wiki prompt — the "Output format" instruction block.
    Returns the active override or the built-in default."""
    return _WIKI_TAIL_OVERRIDE or _DEFAULT_WIKI_TAIL


def wiki_prompt_tail() -> str:
    """Public alias for the active wiki-tail prompt (default or override)."""
    return _wiki_output_format()


def build_wiki(
    cfg: Config,
    db: GraphDB,
    llm: LLMClient | None = None,
    only_module: str | None = None,
    progress=None,
    force: bool = False,
    depth: int = 12,
    cancel_token=None,
    progress_cb=None,
    fetch_links: bool = True,
    force_fetch: bool = False,
) -> list[WikiPage]:
    """Generate (or re-generate) wiki pages for every module (one per
    directory, capped at `depth` directory levels). Saves to
    `<cfg.docgraph_dir>/wiki/`. Returns the list of pages.

    Resumable: modules whose `<slug>.md` already exists on disk with
    non-empty content are skipped (no LLM call). Pass `force=True` to
    rebuild every page from scratch.

    `depth=1` = top-level dirs only (old behavior). `depth=12` (default)
    = one page per leaf folder for any reasonable repo.
    """
    # Pre-fetch stale external links so fresh HTML pages land in the graph
    # the next time the user runs index. Wiki works from the existing DB, so
    # the fetched pages don't affect this wiki run — they keep links fresh
    # for the subsequent index → wiki pipeline.
    try:
        from docgraph.index import _wire_extra_paths
        _wire_extra_paths(cfg)
    except Exception as _we:
        log.debug("wiki extra-paths wiring skipped: %s", _we)
    if fetch_links:
        try:
            from docgraph.index import _maybe_fetch_links
            _maybe_fetch_links(cfg, force=force_fetch)
        except Exception as _fe:
            log.debug("wiki pre-fetch skipped: %s", _fe)

    llm = llm or LLMClient(LLMConfig(
        host=getattr(cfg, "llm_host", "localhost"),
        port=int(getattr(cfg, "llm_port", 1235) or 1235),
        model=getattr(cfg, "llm_model", "qwen3.6-35b") or "qwen3.6-35b",
        format=getattr(cfg, "llm_format", "openai") or "openai",
        max_tokens=int(getattr(cfg, "llm_max_tokens", 150) or 150),
        api_key=getattr(cfg, "llm_api_key", "") or None,
        timeout=int(getattr(cfg, "llm_timeout", 1800) or 1800),
    ))
    # Wiki pages need much more headroom than docstrings (150 tokens).
    # Bump for the wiki call only — the original LLMConfig stays unchanged
    # for any other caller sharing the same client instance. Honors the
    # per-host override if Config carries one (telecode forwards
    # `--llm-max-tokens-wiki` through `host`).
    wiki_budget = int(getattr(cfg, "llm_max_tokens_wiki", 4096) or 4096)
    if llm.cfg.max_tokens < wiki_budget:
        llm.cfg.max_tokens = wiki_budget
    wiki_dir = cfg.data_dir / WIKI_DIRNAME
    wiki_dir.mkdir(parents=True, exist_ok=True)

    groups = _module_groupings(db, depth=depth)
    if only_module:
        groups = [g for g in groups if g["module"] == only_module]
        if not groups:
            log.warning("wiki: no module matched %r", only_module)
            return []

    def _emit(phase: str, current: int = 0, total: int = 0,
              module: str = "") -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(phase, current, total, module)
        except Exception:
            log.debug("wiki progress_cb raised; ignoring", exc_info=True)

    _emit("start", 0, len(groups))
    
    pages: list[WikiPage] = []
    for i, g in enumerate(groups):
        # Cancel checkpoint between modules. Each page is its own LLM
        # call (~5-30s with reasoning models); checking here means a
        # cancel lands on the next module boundary, not after the whole
        # repo. Already-written pages stay on disk → resumable.
        if cancel_token is not None:
            cancel_token.raise_if_set()
        module = g["module"]
        _emit("module", i, len(groups), module)
        slug = _slugify(module)
        title = module if module != "(root)" else "Repository root"
        body_path = wiki_dir / f"{slug}.md"
        facts_path = wiki_dir / f"{slug}.facts.json"

        # Resume: reuse existing page only if it's non-empty AFTER strip.
        # Whitespace-only files (the old `_clean`-truncation bug) must be
        # treated as missing and regenerated.
        existing_body = ""
        if not force and body_path.exists() and body_path.stat().st_size > 0:
            try:
                existing_body = body_path.read_text(encoding="utf-8")
            except Exception:
                existing_body = ""
        if not force and existing_body.strip():
            try:
                facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {"module": module, "file_count": len(g["files"])}
                summary = existing_body.strip().splitlines()[0][:200]
                pages.append(WikiPage(
                    slug=slug, title=title, module=module,
                    summary=summary, body_md=existing_body, facts=facts,
                ))
                if progress:
                    try:
                        progress(i, len(groups), f"{module} (cached)")
                    except Exception:
                        pass
                continue
            except Exception as e:
                log.debug("wiki: failed to reuse %s, regenerating: %s", slug, e)

        if progress:
            try:
                progress(i, len(groups), module)
            except Exception:
                pass
        facts = _facts_for_module(db, module, g["files"])
        prompt = _wiki_prompt(facts)
        # Use chat() not _call_openai — _call_openai routes through _clean()
        # which truncates to a single line (correct for one-sentence
        # docstrings, fatal for multi-paragraph wiki pages).
        # cfg.llm_wiki=False forces the fact-sheet path even if the LLM is
        # reachable — useful when the user wants docstring LLM but not wiki LLM.
        body = llm.chat(prompt) if getattr(cfg, "llm_wiki", True) else ""
        if not body or not body.strip():
            # Fall back to a plain rendering of the facts so the wiki is never blank.
            body = _facts_to_markdown(facts)
            summary = f"{facts['file_count']} files in {module}."
        else:
            summary = body.strip().splitlines()[0][:200]
        page = WikiPage(
            slug=slug, title=title, module=module,
            summary=summary, body_md=body, facts=facts,
        )
        body_path.write_text(body, encoding="utf-8")
        facts_path.write_text(
            json.dumps(facts, indent=2, default=str), encoding="utf-8"
        )
        pages.append(page)

    # Index file. Merge with any existing entries when scoped to a single
    # module so a `--module X` rebuild doesn't wipe other pages from the index.
    new_entries = {
        p.slug: {"slug": p.slug, "title": p.title, "module": p.module, "summary": p.summary}
        for p in pages
    }
    index_path = wiki_dir / INDEX_FILE
    if only_module and index_path.exists():
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            merged = {e["slug"]: e for e in existing if isinstance(e, dict) and e.get("slug")}
            merged.update(new_entries)
            index_payload = list(merged.values())
        except Exception:
            index_payload = list(new_entries.values())
    else:
        index_payload = list(new_entries.values())
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    _emit("done", len(pages), len(pages))
    return pages


def _facts_to_markdown(facts: dict) -> str:
    """Plain Markdown rendering of the fact sheet — used when the LLM is
    unavailable so the wiki always has *something* to show."""
    out = [f"# {facts['module']}", ""]
    out.append(f"_{facts['file_count']} files_")
    out.append("")
    if facts.get("top_classes"):
        out.append("## Key classes")
        for c in facts["top_classes"][:6]:
            lines = (c.get("doc") or "").splitlines()
            doc = (lines[0][:160] if lines else "")
            out.append(f"- **{c.get('name','')}** — `{c.get('file','')}:{c.get('line') or 0}`{(' — ' + doc) if doc else ''}")
        out.append("")
    if facts.get("top_functions"):
        out.append("## Key functions")
        for f in facts["top_functions"][:8]:
            lines = (f.get("doc") or "").splitlines()
            doc = (lines[0][:160] if lines else "")
            out.append(f"- **{f.get('name','')}** — `{f.get('file','')}:{f.get('line') or 0}`{(' — ' + doc) if doc else ''}")
        out.append("")
    if facts.get("importers"):
        out.append("## Used by")
        for i in facts["importers"][:8]:
            out.append(f"- `{i.get('file','')}`")
    return "\n".join(out)


def list_wiki(cfg: Config) -> list[dict]:
    """Read the on-disk wiki index. Empty list if the wiki hasn't been built."""
    idx = cfg.data_dir / WIKI_DIRNAME / INDEX_FILE
    if not idx.exists():
        return []
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return []


def get_wiki_page(cfg: Config, slug: str) -> dict | None:
    """Read one wiki page's Markdown + facts."""
    safe = _slugify(slug)
    if safe != slug:
        return None
    body = cfg.data_dir / WIKI_DIRNAME / f"{slug}.md"
    facts = cfg.data_dir / WIKI_DIRNAME / f"{slug}.facts.json"
    if not body.exists():
        return None
    out = {
        "slug": slug,
        "body_md": body.read_text(encoding="utf-8"),
    }
    if facts.exists():
        try:
            out["facts"] = json.loads(facts.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out
