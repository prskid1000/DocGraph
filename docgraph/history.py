"""Temporal history for symbols.

Index time (bounded, incremental): one `git blame --line-porcelain` per
*changed* file gives every line's last commit + author time. A symbol's

    first_seen_commit   = commit of its definition line (oldest line as fallback)
    last_changed_commit = newest commit among its lines

Lines not yet committed blame to the all-zero sha and are reported as
"uncommitted"; the indexer remembers those files and re-blames them once
HEAD moves (their content hash does not change on commit, so the normal
delta would never revisit them).

Query time: `symbol_log()` runs `git log -L <start>,<end>:<file>` for one
symbol -- the precise per-symbol history, too slow to do for every symbol
at index time but fine for one on demand.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docgraph.proc_util import NO_WINDOW

UNCOMMITTED = "uncommitted"
_ZERO = "0" * 40


def _git(args: list[str], cwd: Path, timeout: float = 60.0) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            errors="replace", timeout=timeout, creationflags=NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def is_git_repo(root: Path) -> bool:
    if git_dir(root) is not None:
        return True
    out = _git(["rev-parse", "--is-inside-work-tree"], root, timeout=10)
    return bool(out and out.strip() == "true")


def git_dir(root: Path) -> Path | None:
    """The .git directory of the work tree containing `root` (a `.git`
    file of a worktree / submodule is followed), or None."""
    p = Path(root).resolve()
    for d in (p, *p.parents):
        g = d / ".git"
        try:
            if g.is_dir():
                return g
            if g.is_file():
                txt = g.read_text(encoding="utf-8", errors="replace").strip()
                if txt.startswith("gitdir:"):
                    q = Path(txt[7:].strip())
                    return q if q.is_absolute() else (d / q).resolve()
        except OSError:
            return None
    return None


def head_fast(root: Path) -> str | None:
    """HEAD's commit read from the .git files (no subprocess): HEAD ->
    loose ref -> packed-refs. None when it cannot tell (the caller falls
    back to `git rev-parse`)."""
    gd = git_dir(root)
    if gd is None:
        return None
    try:
        h = (gd / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not h.startswith("ref:"):
        return h if len(h) >= 40 else None
    ref = h[4:].strip()
    common = gd
    try:
        cd = gd / "commondir"
        if cd.exists():
            c = Path(cd.read_text(encoding="utf-8").strip())
            common = c if c.is_absolute() else (gd / c).resolve()
    except OSError:
        pass
    for base in (gd, common):
        f = base / ref
        try:
            if f.is_file():
                v = f.read_text(encoding="utf-8").strip()
                return v or None
        except OSError:
            continue
    try:
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref) and not line.startswith(("#", "^")):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


def head(root: Path) -> str | None:
    fast = head_fast(root)
    if fast:
        return fast
    out = _git(["rev-parse", "HEAD"], root, timeout=10)
    return out.strip() if out and out.strip() else None


def blame_file(root: Path, rel_to_root: str) -> list[tuple[str, int]] | None:
    """[(sha, author_time)] per line (index 0 = line 1), or None if the
    file is not tracked / git failed."""
    out = _git(["blame", "--line-porcelain", "--", rel_to_root], root)
    if not out:
        return None
    lines: list[tuple[str, int]] = []
    cur_sha = ""
    cur_ts = 0
    meta: dict[str, int] = {}
    expect_header = True
    for raw in out.splitlines():
        if raw.startswith("\t"):
            if cur_sha:
                ts = cur_ts or meta.get(cur_sha, 0)
                meta.setdefault(cur_sha, ts)
                lines.append((UNCOMMITTED if cur_sha == _ZERO else cur_sha[:12], ts))
            expect_header = True
            continue
        if expect_header:
            parts = raw.split(" ")
            if parts and len(parts[0]) == 40:
                cur_sha = parts[0]
                cur_ts = meta.get(cur_sha, 0)
                expect_header = False
                continue
        if raw.startswith("author-time "):
            try:
                cur_ts = int(raw.split(" ", 1)[1])
            except ValueError:
                cur_ts = 0
    return lines


def symbol_span_history(blame: list[tuple[str, int]], start: int, end: int) -> dict:
    """first/last commit info for lines [start, end] (1-based)."""
    if not blame:
        return {}
    s = max(1, start)
    e = min(len(blame), max(start, end))
    span = blame[s - 1:e]
    if not span:
        return {}
    first_sha, first_ts = blame[s - 1]
    committed = [x for x in span if x[0] != UNCOMMITTED]
    if first_sha == UNCOMMITTED and committed:
        first_sha, first_ts = min(committed, key=lambda t: t[1])
    if any(x[0] == UNCOMMITTED for x in span):
        last_sha, last_ts = UNCOMMITTED, max((x[1] for x in span), default=0)
    else:
        top = max(t[1] for t in span)
        newest = [t for t in span if t[1] == top]
        # Same-second commits tie on author time; a commit other than the
        # one that introduced the symbol is the later one.
        last_sha, last_ts = next((t for t in newest if t[0] != first_sha), newest[0])
    return {"fc": first_sha, "fts": int(first_ts), "lc": last_sha, "lts": int(last_ts)}


def blame_many(jobs: list[tuple[str, Path, str]], workers: int = 8,
               max_files: int = 5000) -> dict[str, list[tuple[str, int]]]:
    """jobs: [(logical_rel, root, rel_to_root)] -> {logical_rel: blame}."""
    jobs = jobs[:max_files]
    out: dict[str, list[tuple[str, int]]] = {}
    if not jobs:
        return out

    def _one(job):
        logical, root, rel = job
        return logical, blame_file(root, rel)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for logical, b in ex.map(_one, jobs):
            if b:
                out[logical] = b
    return out


def symbol_log(root: Path, rel_to_root: str, start: int, end: int,
               limit: int = 10) -> list[dict]:
    """`git log -L` for one symbol's current line range."""
    out = _git(["log", f"-{int(limit)}", "-L", f"{int(start)},{int(end)}:{rel_to_root}",
                "--format=__C__%h|%an|%ad|%s", "--date=short"], root, timeout=60)
    rows: list[dict] = []
    if not out:
        return rows
    for line in out.splitlines():
        if not line.startswith("__C__"):
            continue
        parts = line[5:].split("|", 3)
        if len(parts) == 4:
            rows.append({"commit": parts[0], "author": parts[1],
                         "date": parts[2], "subject": parts[3]})
    return rows
