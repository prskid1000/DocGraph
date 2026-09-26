"""Pure helpers behind the analysis tools (no Cypher, no I/O).

retrieve.py fetches rows from Kuzu and delegates the algorithmic parts
here: unified-diff parsing, explainable risk scoring, token budgeting,
Aider-style repo-map rendering, signature extraction, and the graph
metrics behind `health`.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

# ---- tokens -----------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """~4 chars per token -- close enough for budget trimming across the
    usual BPE tokenizers, and dependency-free."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


# ---- unified diff -----------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(text: str) -> list[dict]:
    """Parse a unified diff (git or plain) into per-file change records:
    [{path, old_path, status, ranges, old_ranges, added, removed, diff}]
    `ranges` are 1-based inclusive line ranges on the NEW side; a pure
    deletion hunk contributes its anchor line so the enclosing symbol is
    still found."""
    files: list[dict] = []
    cur: dict | None = None

    def _start(old: str | None, new: str | None) -> dict:
        rec = {"path": new or old or "", "old_path": old or "", "status": "modified",
               "ranges": [], "old_ranges": [], "added": 0, "removed": 0, "diff": []}
        files.append(rec)
        return rec

    old_name: str | None = None
    rem_o = rem_n = 0
    for line in (text or "").splitlines():
        if (rem_o > 0 or rem_n > 0) and cur is not None:
            # Inside a hunk: content lines only (a removed "-- x" line looks
            # like a file header otherwise).
            tag = line[:1]
            if tag == "\\":
                cur["diff"].append(line)
                continue
            if tag == "-":
                rem_o -= 1
                cur["removed"] += 1
            elif tag == "+":
                rem_n -= 1
                cur["added"] += 1
            else:
                rem_o -= 1
                rem_n -= 1
            cur["diff"].append(line)
            continue
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            cur = _start(m.group(1) if m else None, m.group(2) if m else None)
            cur["diff"].append(line)
            old_name = None
            continue
        if line.startswith("--- "):
            p = line[4:].strip()
            old_name = None if p == "/dev/null" else re.sub(r"^a/", "", p.split("\t")[0])
            if cur is None or cur.get("_seen_hunk"):
                cur = _start(old_name, None)
            if p == "/dev/null" and cur is not None:
                cur["status"] = "added"
            if cur is not None:
                cur["diff"].append(line)
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            if cur is None:
                cur = _start(old_name, None)
            if p == "/dev/null":
                cur["status"] = "deleted"
                cur["path"] = cur["old_path"] or cur["path"]
            else:
                cur["path"] = re.sub(r"^b/", "", p.split("\t")[0])
            cur["diff"].append(line)
            continue
        if cur is None:
            continue
        if line.startswith("new file mode"):
            cur["status"] = "added"
        elif line.startswith("deleted file mode"):
            cur["status"] = "deleted"
        m = _HUNK.match(line)
        if m:
            cur["_seen_hunk"] = True
            os_ = int(m.group(1))
            oc = int(m.group(2)) if m.group(2) is not None else 1
            ns = int(m.group(3))
            nc = int(m.group(4)) if m.group(4) is not None else 1
            cur["ranges"].append((max(ns, 1), max(ns, 1) + max(nc - 1, 0)))
            cur["old_ranges"].append((max(os_, 1), max(os_, 1) + max(oc - 1, 0)))
            rem_o, rem_n = oc, nc
        cur["diff"].append(line)
    for f in files:
        f.pop("_seen_hunk", None)
        f["diff"] = "\n".join(f["diff"])[:4000]
    return [f for f in files if f["path"]]


def overlaps(s: int, e: int, ranges: list[tuple[int, int]]) -> int:
    """Number of changed lines of [s, e] covered by ranges."""
    n = 0
    for rs, re_ in ranges:
        lo, hi = max(s, rs), min(e, re_)
        if lo <= hi:
            n += hi - lo + 1
    return n


# ---- risk -------------------------------------------------------------------

RISK_LEVELS = ((75, "critical"), (50, "high"), (25, "medium"), (0, "low"))


def risk_level(score: float) -> str:
    for threshold, name in RISK_LEVELS:
        if score >= threshold:
            return name
    return "low"


def symbol_risk(*, pagerank_pct: float, n_callers: int, n_routes: int,
                n_entries: int, n_tests: int, changed_lines: int,
                is_test: bool = False) -> dict:
    """Explainable 0..100 risk for one changed symbol. Every factor reports
    the evidence and the points it contributed."""
    factors: list[dict] = []

    def add(name: str, points: float, detail: str) -> None:
        factors.append({"factor": name, "points": round(points, 1), "detail": detail})

    if is_test:
        add("test_code", 0.0, "the symbol is a test; changing it cannot break callers")
        return {"score": 0.0, "level": "low", "factors": factors}
    add("centrality", 25.0 * max(0.0, min(1.0, pagerank_pct)),
        f"PageRank percentile {pagerank_pct * 100:.0f}")
    add("blast_radius", min(25.0, 5.0 * math.log2(1 + n_callers)),
        f"{n_callers} transitive caller(s)")
    add("entry_points", min(20.0, 7.0 * n_routes + 3.0 * n_entries),
        f"{n_routes} route/tool handler(s), {n_entries} other entry point(s) reach it")
    if n_tests == 0:
        add("test_gap", 20.0, "no test reaches it")
    elif n_tests <= 2:
        add("test_gap", 8.0, f"only {n_tests} test(s) reach it")
    else:
        add("test_gap", 0.0, f"{n_tests} tests reach it")
    add("change_size", min(10.0, changed_lines / 5.0), f"{changed_lines} changed line(s)")
    score = round(min(100.0, sum(f["points"] for f in factors)), 1)
    return {"score": score, "level": risk_level(score), "factors": factors}


def overall_risk(per_symbol: list[dict]) -> dict:
    """Max symbol risk plus a small share of the rest, capped at 100."""
    scores = sorted((s["score"] for s in per_symbol), reverse=True)
    if not scores:
        return {"score": 0.0, "level": "low",
                "summary": "no indexed symbol overlaps the diff"}
    total = min(100.0, scores[0] + 0.1 * sum(scores[1:]))
    return {"score": round(total, 1), "level": risk_level(total),
            "summary": f"{len(scores)} changed symbol(s); riskiest scores {scores[0]:.0f}"}


# ---- signatures + repo map --------------------------------------------------

def signature_of(body: str, fallback: str = "", max_lines: int = 6) -> str:
    """Declaration line(s) of a function/class, whitespace-collapsed:
    everything up to the line ending the header (`:` for Python, `{` for
    brace languages), at most max_lines."""
    if not body:
        return fallback.strip()
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines[:max_lines]:
        s = ln.rstrip()
        if s.lstrip().startswith("@"):
            continue
        out.append(s.strip())
        if s.endswith(":") or s.endswith("{") or s.endswith(";") or s.endswith("=>"):
            break
    sig = " ".join(out)
    sig = re.sub(r"\s+", " ", sig).strip()
    if sig.endswith("{"):
        sig = sig[:-1].rstrip()
    return sig[:240] or fallback.strip()


def render_repo_map(entries: list[dict]) -> str:
    """entries: [{file, line, kind, signature, parent}] (already chosen).
    Groups by file, orders by line, indents methods under their class
    (Aider-style, ASCII only)."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_file[e["file"]].append(e)
    parts: list[str] = []
    for f in sorted(by_file):
        rows = sorted(by_file[f], key=lambda e: e.get("line") or 0)
        parts.append(f"{f}:")
        prev_line = None
        for e in rows:
            if prev_line is not None and (e.get("line") or 0) - prev_line > 1:
                parts.append("|...")
            indent = "    " if e.get("parent") else ""
            parts.append(f"|{indent}{e['signature']}")
            prev_line = e.get("line_end") or e.get("line") or 0
        parts.append("")
    return "\n".join(parts).rstrip() + ("\n" if parts else "")


def fit_to_budget(ranked: list[dict], tokens: int, render) -> tuple[str, int]:
    """Binary-search the largest prefix of `ranked` whose rendering fits in
    `tokens`. Returns (text, n_included)."""
    if tokens <= 0 or not ranked:
        return "", 0
    lo, hi = 0, len(ranked)
    best_text, best_n = "", 0
    while lo <= hi:
        mid = (lo + hi) // 2
        text = render(ranked[:mid]) if mid else ""
        if estimate_tokens(text) <= tokens:
            best_text, best_n = text, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text, best_n


# ---- token-budgeted sections -------------------------------------------------

def trim_sections(sections: list[tuple[str, list[str]]], tokens: int,
                  min_items: int = 1) -> tuple[str, dict[str, int], list[str]]:
    """sections: [(title, [line, ...])] in priority order. Drops trailing
    items from the lowest-priority sections first until the rendering fits.
    Returns (markdown, kept_counts, truncated_titles)."""
    keep = {t: len(items) for t, items in sections}

    def render() -> str:
        out: list[str] = []
        for t, items in sections:
            n = keep[t]
            if n <= 0:
                continue
            out.append(f"## {t}")
            out.extend(items[:n])
            if n < len(items):
                out.append(f"... ({len(items) - n} more)")
            out.append("")
        return "\n".join(out).strip() + "\n"

    text = render()
    truncated: list[str] = []
    order = [t for t, _ in sections][::-1]  # lowest priority first
    while estimate_tokens(text) > tokens:
        progressed = False
        for t in order:
            floor = min_items if t == sections[0][0] else 0
            if keep[t] > floor:
                keep[t] = max(floor, keep[t] - max(1, keep[t] // 3))
                if t not in truncated:
                    truncated.append(t)
                progressed = True
                break
        if not progressed:
            break
        text = render()
    return text, keep, truncated
