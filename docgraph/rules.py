"""Auto-attach context rules.

Cursor's `.cursor/rules/*.mdc` files specify per-glob context that gets
injected when the user opens a matching file. We read the same format —
plus `AGENTS.md` and Claude's `CLAUDE.md` — and expose them via MCP so any
agent can ask "what rules apply to this file?".

MDC frontmatter (subset we care about):
    ---
    description: short summary
    globs: ["src/**/*.test.ts", "tests/**/*.py"]
    alwaysApply: false
    ---
    <body>
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from docgraph.config import Config

log = logging.getLogger(__name__)


@dataclass
class Rule:
    name: str            # filename or section
    source: str          # absolute path
    description: str = ""
    globs: list[str] = field(default_factory=list)
    always_apply: bool = False
    body: str = ""


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tiny frontmatter parser. Doesn't pull in PyYAML; handles the subset
    Cursor actually uses (key: value, value lists, simple booleans)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_block, body = m.group(1), m.group(2)
    out: dict = {}
    for line in fm_block.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        elif v.startswith("["):
            # ["a", "b"] or [a, b]
            inside = v.strip("[]").strip()
            if inside:
                items = [it.strip().strip("'\"") for it in inside.split(",")]
                out[k] = [it for it in items if it]
            else:
                out[k] = []
        elif v:
            out[k] = v.strip("'\"")
    return out, body


def _load_mdc(path: Path) -> Rule | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm, body = _parse_frontmatter(text)
    globs_raw = fm.get("globs", [])
    if isinstance(globs_raw, str):
        # Some users write `globs: src/**/*.py` unquoted
        globs_raw = [g.strip() for g in globs_raw.split(",") if g.strip()]
    return Rule(
        name=path.stem,
        source=str(path),
        description=str(fm.get("description", "")),
        globs=list(globs_raw),
        always_apply=bool(fm.get("alwaysApply") or fm.get("always_apply")),
        body=body.strip(),
    )


def collect_rules(cfg: Config) -> list[Rule]:
    """Scan all configured roots for .cursor/rules/*.mdc, AGENTS.md,
    and CLAUDE.md (project-level). Returns a flat list."""
    out: list[Rule] = []
    for root, _prefix in cfg.roots_with_prefix():
        # Cursor MDC rules
        rules_dir = root / ".cursor" / "rules"
        if rules_dir.is_dir():
            for p in sorted(rules_dir.rglob("*.mdc")):
                r = _load_mdc(p)
                if r is not None:
                    out.append(r)
        # Agent files at the repo root — always-on docs
        for fname in ("AGENTS.md", "CLAUDE.md"):
            p = root / fname
            if p.is_file():
                try:
                    body = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                out.append(Rule(
                    name=fname,
                    source=str(p),
                    description=f"{fname} repo-level guidance",
                    globs=["**/*"],
                    always_apply=True,
                    body=body.strip(),
                ))
    return out


def rules_for(cfg: Config, file_path: str) -> list[dict]:
    """Return the rules whose globs match the given file. Always-apply rules
    are returned first regardless of glob."""
    rules = collect_rules(cfg)
    out: list[dict] = []
    for r in rules:
        matched = r.always_apply
        if not matched and r.globs:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", r.globs)
            matched = spec.match_file(file_path)
        if matched:
            out.append({
                "name": r.name,
                "source": r.source,
                "description": r.description,
                "globs": r.globs,
                "always_apply": r.always_apply,
                "body": r.body,
            })
    # Always-apply first, then specificity (more globs = less specific = lower)
    out.sort(key=lambda x: (not x["always_apply"], len(x["globs"]) or 99))
    return out
