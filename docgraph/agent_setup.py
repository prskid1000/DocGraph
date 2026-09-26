"""`docgraph agent-setup`: write agent skill files + hooks into a repo.

Only ever runs when a user invokes it explicitly, is idempotent (a second
run reports every file "unchanged"), and has --dry-run. What it writes:

  claude   .claude/skills/docgraph/SKILL.md          how to use the tools
           .claude/skills/docgraph-area-<slug>/SKILL.md   one per cluster
           .claude/settings.json  PreToolUse hook (Edit|Write|MultiEdit)
                                  -> `docgraph hook pre-edit` impact hint
  codex    AGENTS.md  (a marked block)  +  .agents/skills/docgraph*/SKILL.md
  agy      same files as codex (agy reads AGENTS.md and .agents/skills)
  git      .git/hooks/post-commit (a marked block) -> `docgraph hook
           post-commit` stale-index notice (with --hooks)

Existing content is preserved: JSON is merged (only our hook entries are
replaced), Markdown blocks live between markers, git hooks get a marked
block appended.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from docgraph.proc_util import NO_WINDOW

BEGIN = "<!-- docgraph:begin -->"
END = "<!-- docgraph:end -->"
GIT_BEGIN = "# >>> docgraph post-commit >>>"
GIT_END = "# <<< docgraph post-commit <<<"
HOOK_MARK = "docgraph hook pre-edit"

TARGETS = ("claude", "codex", "agy")


@dataclass
class Action:
    path: Path
    content: str
    kind: str = "file"      # file | block | json_hooks | git_hook
    status: str = ""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:48] or "area"


def fetch_json(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _q(host_url: str, path: str, root: str | None, **params) -> str:
    if root:
        params["root"] = root
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{host_url.rstrip('/')}{path}" + (f"?{qs}" if qs else "")


# ---- content -----------------------------------------------------------------

def main_skill(host_url: str, root: str | None) -> str:
    root_arg = f', root="{root}"' if root else ""
    return f"""---
name: docgraph
description: Code knowledge graph for this repo (DocGraph). Use before editing or reviewing code to find definitions, callers, blast radius, tests to run, routes and architecture instead of grepping.
---

# DocGraph

A DocGraph host indexes this repository ({host_url}). Its MCP tools (server
`docgraph`, or REST at {host_url}/api/...) answer structural questions
directly:

| Question | Tool |
|---|---|
| Everything about one symbol (callers, callees, tests, flows, history) | `context(symbol{root_arg}, tokens=2000)` |
| What does my diff affect / which tests to run / how risky | `detect_changes(ref=None{root_arg})` or `detect_changes(diff=...)` |
| Map of the relevant area in N tokens | `repo_map(focus=["path/or/symbol"], tokens=1024)` |
| Who calls X, transitively | `impact_of(target, min_confidence=0.5)` |
| How does A reach B | `trace(a, b)` |
| HTTP routes / MCP tools and what they touch | `route_map()`, `api_impact("GET /x")` |
| Architecture | `list_clusters()`, `cluster(id)` |
| Dead code, hubs, cycles, untested hotspots | `health()` |
| Rename plan (never writes) | `rename(symbol, new_name)` |

Edges carry a resolution `confidence` (0.95 import map ... 0.3 fuzzy);
pass `min_confidence=0.5` when you need certainty. Ambiguous call sites are
kept as candidates, not invented edges.

Before editing a symbol: call `context`. Before committing: call
`detect_changes` and run its `test_command`.
"""


def area_skill(c: dict, host_url: str) -> str:
    members = ", ".join(f"`{m}`" for m in (c.get("top_members") or [])[:8])
    files = "\n".join(f"- `{f}`" for f in (c.get("files") or [])[:8])
    return f"""---
name: docgraph-area-{_slug(c['name'])}
description: Working in the "{c['name']}" area of this repo ({c.get('size', '?')} symbols; key members {', '.join((c.get('top_members') or [])[:4])}). Load when editing these files.
---

# Area: {c['name']}

Community #{c['id']} detected by DocGraph (cohesion {c.get('cohesion', 0)}).

Key symbols: {members or '-'}

Main files:
{files or '- (none)'}

Use `cluster(id={c['id']})` for the full member list, its API (members
called from outside) and the areas it depends on; `context(symbol)` before
changing one of the key symbols.
"""


def agents_block(host_url: str, root: str | None, clusters: list[dict]) -> str:
    lines = [BEGIN, "## DocGraph", "",
             f"This repo is indexed by DocGraph ({host_url}). Prefer its tools over grep for "
             "structure: `context(symbol)` before editing, `detect_changes()` before "
             "committing (runs its `test_command`), `impact_of`, `trace`, `repo_map`, "
             "`route_map`, `health`. See `.agents/skills/docgraph/SKILL.md`."]
    if clusters:
        lines += ["", "Areas:"]
        for c in clusters[:12]:
            lines.append(f"- {c['name']} ({c.get('size', '?')} symbols) -> "
                         f"`.agents/skills/docgraph-area-{_slug(c['name'])}/SKILL.md`")
    lines.append(END)
    return "\n".join(lines) + "\n"


def _cmd(docgraph_cmd: str, sub: str, host_url: str, root: str | None) -> str:
    parts = [docgraph_cmd, "hook", sub, "--host-url", host_url]
    if root:
        parts += ["--root", root]
    return " ".join(parts)


# ---- planning / applying -----------------------------------------------------

def plan(repo: Path, targets: list[str], host_url: str, root: str | None,
         clusters: list[dict], hooks: bool = True, per_cluster: bool = True,
         docgraph_cmd: str = "docgraph", max_clusters: int = 12) -> list[Action]:
    acts: list[Action] = []
    tset = set(targets)
    chosen = [c for c in clusters if c.get("size", 0) >= 3][:max_clusters] if per_cluster else []
    if "claude" in tset:
        base = repo / ".claude" / "skills"
        acts.append(Action(base / "docgraph" / "SKILL.md", main_skill(host_url, root)))
        for c in chosen:
            acts.append(Action(base / f"docgraph-area-{_slug(c['name'])}" / "SKILL.md",
                               area_skill(c, host_url)))
        if hooks:
            acts.append(Action(repo / ".claude" / "settings.json",
                               _cmd(docgraph_cmd, "pre-edit", host_url, root), kind="json_hooks"))
    if tset & {"codex", "agy"}:
        base = repo / ".agents" / "skills"
        acts.append(Action(base / "docgraph" / "SKILL.md", main_skill(host_url, root)))
        for c in chosen:
            acts.append(Action(base / f"docgraph-area-{_slug(c['name'])}" / "SKILL.md",
                               area_skill(c, host_url)))
        acts.append(Action(repo / "AGENTS.md", agents_block(host_url, root, chosen), kind="block"))
    if hooks:
        hook_dir = _git_hooks_dir(repo)
        if hook_dir is not None:
            acts.append(Action(hook_dir / "post-commit",
                               _cmd(docgraph_cmd, "post-commit", host_url, root) + " || true",
                               kind="git_hook"))
    return acts


def _git_hooks_dir(repo: Path) -> Path | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=repo,
                             capture_output=True, text=True, timeout=10,
                             creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    p = Path(out.stdout.strip())
    return p if p.is_absolute() else (repo / p)


def _render(a: Action) -> str:
    """Final content for an action, merged with what is on disk."""
    old = a.path.read_text(encoding="utf-8") if a.path.exists() else None
    if a.kind == "file":
        return a.content
    if a.kind == "block":
        if old is None:
            return a.content
        if BEGIN in old and END in old:
            pre, rest = old.split(BEGIN, 1)
            _mid, post = rest.split(END, 1)
            return pre + a.content.rstrip("\n") + post
        sep = "" if old.endswith("\n\n") else ("\n" if old.endswith("\n") else "\n\n")
        return old + sep + a.content
    if a.kind == "json_hooks":
        try:
            data = json.loads(old) if old else {}
        except ValueError:
            raise ValueError(f"{a.path} is not valid JSON; refusing to merge")
        hooks = data.setdefault("hooks", {})
        pre = [h for h in hooks.get("PreToolUse", [])
               if not any(HOOK_MARK in (x.get("command") or "") for x in h.get("hooks", []))]
        pre.append({"matcher": "Edit|Write|MultiEdit",
                    "hooks": [{"type": "command", "command": a.content, "timeout": 10}]})
        hooks["PreToolUse"] = pre
        return json.dumps(data, indent=2) + "\n"
    if a.kind == "git_hook":
        block = f"{GIT_BEGIN}\n{a.content}\n{GIT_END}\n"
        if old is None:
            return "#!/bin/sh\n" + block
        if GIT_BEGIN in old and GIT_END in old:
            pre, rest = old.split(GIT_BEGIN, 1)
            _mid, post = rest.split(GIT_END, 1)
            return pre + block.rstrip("\n") + post
        return old + ("" if old.endswith("\n") else "\n") + block
    raise ValueError(a.kind)


def apply(actions: list[Action], dry_run: bool = False) -> list[dict]:
    out = []
    for a in actions:
        try:
            new = _render(a)
        except ValueError as exc:
            out.append({"path": str(a.path), "status": "error", "detail": str(exc)})
            continue
        old = a.path.read_text(encoding="utf-8") if a.path.exists() else None
        status = "unchanged" if old == new else ("updated" if old is not None else "created")
        if status != "unchanged" and not dry_run:
            a.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.path.with_name(a.path.name + ".docgraph.tmp")
            tmp.write_text(new, encoding="utf-8", newline="\n")
            tmp.replace(a.path)
            if a.kind == "git_hook":
                try:
                    a.path.chmod(0o755)
                except OSError:
                    pass
        out.append({"path": str(a.path), "status": ("would be " + status) if dry_run and status != "unchanged"
                    else status, "kind": a.kind})
    return out


# ---- hook runtime --------------------------------------------------------------

def pre_edit_hint(payload: dict, host_url: str, root: str | None) -> str | None:
    """Impact hint for the file an agent is about to edit. Never raises."""
    ti = payload.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("path") or ""
    if not fp:
        return None
    roots = fetch_json(f"{host_url.rstrip('/')}/api/roots") or []
    rel = None
    slug = root
    for r in roots:
        try:
            rp = Path(r["path"]).resolve()
            rel = Path(fp).resolve().relative_to(rp).as_posix()
            slug = slug or r["slug"]
            break
        except Exception:
            continue
    if rel is None:
        return None
    imp = fetch_json(_q(host_url, "/api/impact_of", slug, target=rel, depth=2, limit=20)) or {}
    callers = imp.get("callers") or []
    importers = imp.get("importers") or []
    tests = imp.get("tests") or []
    if not (callers or importers or tests):
        return None
    top = ", ".join(sorted({c.get("name") for c in callers if c.get("name")})[:6])
    t = ", ".join(sorted({x.get("name") for x in tests if x.get("name")})[:5])
    return (f"DocGraph: {rel} has {len(callers)} caller(s) within 2 hops"
            + (f" ({top})" if top else "") + f", {len(importers)} importing file(s)"
            + (f"; tests: {t}" if t else "; no tests found")
            + ". Run detect_changes() after editing.")


def post_commit_notice(host_url: str, root: str | None) -> str:
    roots = fetch_json(f"{host_url.rstrip('/')}/api/roots")
    if roots is None:
        return "DocGraph: host not reachable; run `docgraph index` to refresh the graph."
    slot = next((r for r in roots if not root or r.get("slug") == root), None)
    if slot and slot.get("watching"):
        return f"DocGraph: {slot['slug']} is watched; the index refreshes automatically."
    name = slot["slug"] if slot else (root or "?")
    return (f"DocGraph: new commit -- the {name} index is now stale. Reindex from the UI "
            f"(Index page) or `docgraph index`.")
