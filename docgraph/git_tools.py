"""Git-aware retrieval helpers.

Cursor's @Commit / @PR / @Recent Changes / Cursor Blame surface diff- and
attribution-aware context. We expose the same primitives via MCP, but go
further by joining them to the graph: a changed file's entities + their
1-hop neighborhood travel together, so the agent doesn't need to chain
calls.

Pure git CLI shell-out; no GitPython dependency. Read-only.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from docgraph.config import Config
from docgraph.proc_util import NO_WINDOW


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
            errors="replace",
            creationflags=NO_WINDOW,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _changed_files(cfg: Config, ref: str | None) -> list[tuple[Path, str]]:
    """Return [(absolute_path, logical_rel)] of files changed in the diff.

    ref:
      - None      → working-tree changes (unstaged + staged)
      - "HEAD"    → last commit's diff
      - "abc123"  → diff of that commit
      - "main"    → diff of HEAD vs main (current branch's net diff)
    """
    out: list[tuple[Path, str]] = []
    for root, prefix in cfg.roots_with_prefix():
        if ref is None:
            args = ["diff", "--name-only", "HEAD"]
        elif ".." in ref or "..." in ref or ref in ("main", "master"):
            # Branch diff vs base
            args = ["diff", "--name-only", f"{ref}...HEAD"]
        else:
            # Single revision: show files in that commit
            args = ["show", "--name-only", "--pretty=format:", ref]
        text = _git(args, root)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            full = root / line
            if not full.exists():
                # Deleted in this diff — still useful as an "affected file"
                pass
            logical = f"{prefix}{line.replace(chr(92), '/')}"
            out.append((full, logical))
    return out


def _diff_for_file(root: Path, rel: str, ref: str | None) -> str:
    if ref is None:
        return _git(["diff", "HEAD", "--", rel], root)
    if ".." in ref or "..." in ref or ref in ("main", "master"):
        return _git(["diff", f"{ref}...HEAD", "--", rel], root)
    return _git(["show", ref, "--", rel], root)


# Hunk header regex: @@ -a,b +c,d @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def changed_line_ranges(diff_text: str) -> list[tuple[int, int]]:
    """From a unified diff body, extract (start, end) line ranges that were
    added/modified on the new side."""
    out: list[tuple[int, int]] = []
    for m in _HUNK_RE.finditer(diff_text):
        start = int(m.group(1))
        count = int(m.group(2) or "1")
        out.append((start, start + max(count - 1, 0)))
    return out


def changed_entities(cfg: Config, db, ref: str | None) -> dict:
    """Entities (Function/Class) that overlap a changed line range, plus
    the changed file paths themselves and a per-file 1-hop neighborhood
    (callers, callees, importers).

    Returns:
      {
        "ref": <ref>,
        "files": [{"path", "diff_excerpt", "ranges"}],
        "entities": [{"name", "qname", "file", "kind", "line"}],
        "callers_of_changed": [...],   # functions calling into the changed entities
      }
    """
    files_out: list[dict] = []
    file_to_root: dict[str, Path] = {}
    for root, prefix in cfg.roots_with_prefix():
        for path, logical in _changed_files(cfg, ref):
            try:
                rel_to_root = str(path.relative_to(root)).replace("\\", "/")
            except ValueError:
                continue
            if logical.startswith(prefix) or prefix == "":
                diff_text = _diff_for_file(root, rel_to_root, ref)
                ranges = changed_line_ranges(diff_text)
                files_out.append({
                    "path": logical,
                    "ranges": ranges,
                    "diff_excerpt": diff_text[:2000],
                })
                file_to_root[logical] = root

    if not files_out:
        return {"ref": ref, "files": [], "entities": [], "callers_of_changed": []}

    # Find entities overlapping the changed ranges.
    entities: list[dict] = []
    seen_qnames: set[str] = set()
    for fr in files_out:
        path = fr["path"]
        ranges = fr["ranges"]
        if not ranges:
            continue
        for label in ("Function", "Class"):
            try:
                rows = db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.file = $f "
                    f"RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                    f"n.file AS file, n.line_start AS s, n.line_end AS e",
                    {"f": path},
                )
            except Exception:
                rows = []
            for r in rows:
                s, e = r["s"], r["e"]
                for rs, re_ in ranges:
                    if not (e < rs or s > re_):
                        if r["qname"] in seen_qnames:
                            break
                        seen_qnames.add(r["qname"])
                        entities.append({
                            "name": r["name"],
                            "qname": r["qname"],
                            "file": r["file"],
                            "kind": label,
                            "line": s,
                        })
                        break

    # 1-hop callers of the changed entities — what's about to break?
    callers: list[dict] = []
    if entities:
        try:
            rows = db.fetch_all(
                "MATCH (caller:Function)-[:CALLS]->(callee:Function) "
                "WHERE callee.qname IN $qs "
                "RETURN DISTINCT caller.qname AS caller_qname, caller.name AS name, "
                "caller.file AS file, caller.line_start AS line, "
                "callee.qname AS callee_qname",
                {"qs": [e["qname"] for e in entities if e["kind"] == "Function"]},
            )
            callers = rows
        except Exception:
            callers = []

    return {
        "ref": ref,
        "files": files_out,
        "entities": entities,
        "callers_of_changed": callers,
    }


# --- Blame -------------------------------------------------------------


_BLAME_LINE_RE = re.compile(
    r"^(\^?[0-9a-f]+)\s+(?:\S+\s+)?\(([^)]+)\)\s",
)


def blame_lines(cfg: Config, file_path: str, line_start: int = 1, line_end: int | None = None) -> list[dict]:
    """Run `git blame -L start,end -- file` and parse author + commit.

    Returns [{commit, author, date, line, content}]. Commit is the abbrev SHA;
    "^" prefix means the line was untouched in the initial commit.
    """
    # Resolve the logical path back to (root, rel_to_root)
    abs_path: Path | None = None
    rel_to_root: str | None = None
    owning_root: Path | None = None
    for root, prefix in cfg.roots_with_prefix():
        if prefix == "":
            abs_path = root / file_path
            rel_to_root = file_path
            owning_root = root
            break
        if file_path.startswith(prefix):
            rel = file_path[len(prefix):]
            abs_path = root / rel
            rel_to_root = rel
            owning_root = root
            break
    if abs_path is None or owning_root is None or rel_to_root is None:
        return []

    rng = f"{line_start},{line_end}" if line_end else f"{line_start}"
    try:
        out = subprocess.check_output(
            ["git", "blame", "-L", rng, "--date=short", "--", rel_to_root],
            cwd=owning_root,
            text=True,
            stderr=subprocess.DEVNULL,
            errors="replace",
            creationflags=NO_WINDOW,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    rows: list[dict] = []
    for raw in out.splitlines():
        m = _BLAME_LINE_RE.match(raw)
        if not m:
            continue
        commit, attribution = m.group(1), m.group(2).strip()
        # Attribution looks like "Alice 2024-12-01 42)"
        parts = attribution.rsplit(" ", 2)
        author = parts[0] if len(parts) >= 2 else "?"
        date = parts[1] if len(parts) >= 2 else ""
        try:
            line = int(parts[2].rstrip(")")) if len(parts) >= 3 else 0
        except ValueError:
            line = 0
        # Content is everything after the matched prefix
        content_start = m.end()
        content = raw[content_start:]
        rows.append({
            "commit": commit.lstrip("^"),
            "author": author,
            "date": date,
            "line": line,
            "content": content,
        })
    return rows


# --- Recent commits per file ------------------------------------------


def recent_commits(cfg: Config, file_path: str | None = None, limit: int = 20) -> list[dict]:
    """`git log` with one row per commit, optionally scoped to a file."""
    out: list[dict] = []
    for root, prefix in cfg.roots_with_prefix():
        args = ["log", f"-{limit}", "--pretty=format:%h|%an|%ad|%s", "--date=short"]
        scoped_path: str | None = None
        if file_path:
            if prefix and file_path.startswith(prefix):
                scoped_path = file_path[len(prefix):]
            elif not prefix:
                scoped_path = file_path
            else:
                continue
            args += ["--", scoped_path]
        text = _git(args, root)
        for line in text.splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            sha, author, date, subject = parts
            out.append({
                "commit": sha,
                "author": author,
                "date": date,
                "subject": subject,
                "repo": root.name if prefix else "",
            })
    return out[:limit]
