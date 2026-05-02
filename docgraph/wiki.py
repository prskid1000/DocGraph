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
from docgraph.llm import LLMClient, LLMConfig, llm_config_from_env

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


def _module_groupings(db: GraphDB) -> list[dict]:
    """Group all File rows by their top-level directory. Returns one entry
    per group with its file paths."""
    rows = db.fetch_all("MATCH (f:File) RETURN f.path AS path")
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        path = (r.get("path") or "").replace("\\", "/")
        if not path:
            continue
        # Top-level directory or the file itself if it's at the root
        parts = path.split("/")
        key = parts[0] if len(parts) > 1 else "(root)"
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

    # Top classes / functions in this module by PageRank. We pull `body` and
    # use its first non-empty line as a doc snippet — the schema doesn't have
    # a separate docstring column.
    top_classes = _safe(
        "MATCH (c:Class) WHERE c.file IN $files "
        "RETURN c.name AS name, c.qname AS qname, c.file AS file, "
        "c.line_start AS line, coalesce(c.pagerank, 0.0) AS pagerank, "
        "c.body AS body "
        "ORDER BY pagerank DESC LIMIT 8",
        {"files": files},
    )
    top_functions = _safe(
        "MATCH (f:Function) WHERE f.file IN $files "
        "RETURN f.name AS name, f.qname AS qname, f.file AS file, "
        "f.line_start AS line, coalesce(f.pagerank, 0.0) AS pagerank, "
        "f.body AS body "
        "ORDER BY pagerank DESC LIMIT 12",
        {"files": files},
    )
    # Body → first interesting line as a doc-ish snippet
    for r in (*top_classes, *top_functions):
        b = (r.get("body") or "").strip().splitlines()
        snippet = next((ln.strip(' "\'') for ln in b if ln.strip() and not ln.strip().startswith(("def ", "class ", "@"))), "")
        r["doc"] = snippet[:160]
        r.pop("body", None)
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
    """Build a prompt from a fact sheet. Asks for grounded prose."""
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
            doc = (c.get("doc") or "").splitlines()[0][:120]
            parts.append(f"  - `{c.get('name','')}` ({c.get('file','')}:{c.get('line') or 0}) — {doc}")
    if facts.get("top_functions"):
        parts.append("- Top functions (by structural importance):")
        for f in facts["top_functions"][:8]:
            doc = (f.get("doc") or "").splitlines()[0][:120]
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
    parts += [
        "",
        "## Output format",
        "Write a Markdown page with these sections:",
        "1. **Summary** — 2-3 sentences on the module's purpose.",
        "2. **Key entities** — bulleted list of the most important classes/functions and what each is for.",
        "3. **How it's used** — who imports it, in plain language.",
        "Total length: 200-300 words. No code blocks. Do not list every file. Only state what the facts support.",
    ]
    return "\n".join(parts)


def build_wiki(
    cfg: Config,
    db: GraphDB,
    llm: LLMClient | None = None,
    only_module: str | None = None,
    progress=None,
    force: bool = False,
) -> list[WikiPage]:
    """Generate (or re-generate) wiki pages for every top-level module.
    Saves them to `<cfg.docgraph_dir>/wiki/`. Returns the list of pages.

    Resumable: modules whose `<slug>.md` already exists on disk with
    non-empty content are skipped (no LLM call). Pass `force=True` to
    rebuild every page from scratch.
    """
    llm = llm or LLMClient(llm_config_from_env())
    # Wiki pages are 200-300 words; docstring's 150-token default truncates
    # them mid-sentence. Bump for the wiki call only — the original LLMConfig
    # stays unchanged for any other caller of the same client.
    if llm.cfg.max_tokens < 600:
        llm.cfg.max_tokens = 600
    wiki_dir = cfg.data_dir / WIKI_DIRNAME
    wiki_dir.mkdir(parents=True, exist_ok=True)

    groups = _module_groupings(db)
    if only_module:
        groups = [g for g in groups if g["module"] == only_module]
        if not groups:
            log.warning("wiki: no module matched %r", only_module)
            return []

    pages: list[WikiPage] = []
    for i, g in enumerate(groups):
        module = g["module"]
        slug = _slugify(module)
        title = module if module != "(root)" else "Repository root"
        body_path = wiki_dir / f"{slug}.md"
        facts_path = wiki_dir / f"{slug}.facts.json"

        # Resume: existing non-empty page → reuse without re-LLMing.
        if not force and body_path.exists() and body_path.stat().st_size > 0:
            try:
                body = body_path.read_text(encoding="utf-8")
                facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {"module": module, "file_count": len(g["files"])}
                summary = body.splitlines()[0][:200] if body.strip() else f"{len(g['files'])} files in {module}."
                pages.append(WikiPage(
                    slug=slug, title=title, module=module,
                    summary=summary, body_md=body, facts=facts,
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
        body = llm._call_openai(prompt) if llm.cfg.format == "openai" else llm._call_anthropic(prompt)
        if not body:
            # Fall back to a plain rendering of the facts so the wiki is never blank.
            body = _facts_to_markdown(facts)
            summary = f"{facts['file_count']} files in {module}."
        else:
            summary = body.splitlines()[0][:200]
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
            doc = (c.get("doc") or "").splitlines()[0][:160]
            out.append(f"- **{c.get('name','')}** — `{c.get('file','')}:{c.get('line') or 0}`{(' — ' + doc) if doc else ''}")
        out.append("")
    if facts.get("top_functions"):
        out.append("## Key functions")
        for f in facts["top_functions"][:8]:
            doc = (f.get("doc") or "").splitlines()[0][:160]
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
