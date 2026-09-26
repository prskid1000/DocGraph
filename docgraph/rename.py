"""Apply a rename plan (see Retriever.rename_plan).

Only reached from `POST /api/rename` with `dry_run: false` AND
`apply: true`; the MCP tool is plan-only. Every edit is re-verified
against the file on disk (`before` must still equal the line) so a plan
computed against a stale index never clobbers newer code; mismatches are
skipped and reported. Each file is written atomically (tmp + os.replace).
"""
from __future__ import annotations

import os
from collections import defaultdict


def apply_plan(cfg, plan: dict, sources: tuple[str, ...] = ("graph",)) -> dict:
    edits = [e for e in plan.get("edits") or [] if e.get("source") in sources]
    by_file: dict[str, list[dict]] = defaultdict(list)
    for e in edits:
        by_file[e["file"]].append(e)
    written: list[str] = []
    applied = 0
    skipped: list[dict] = []
    for logical, fedits in sorted(by_file.items()):
        full = cfg.path_for(logical).resolve()
        inside = False
        for root, _p in cfg.roots_with_prefix():
            try:
                full.relative_to(root.resolve())
                inside = True
                break
            except ValueError:
                continue
        if not inside or cfg.ai_blocked_logical(logical):
            skipped.extend(dict(e, why="outside repo or ai-blocked") for e in fedits)
            continue
        try:
            raw = full.read_bytes()
        except OSError as exc:
            skipped.extend(dict(e, why=f"unreadable: {exc}") for e in fedits)
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(newline)
        changed = False
        for e in fedits:
            i = int(e["line"]) - 1
            if not (0 <= i < len(lines)) or lines[i] != e["before"]:
                skipped.append(dict(e, why="line changed since the plan was made"))
                continue
            lines[i] = e["after"]
            applied += 1
            changed = True
        if changed:
            tmp = full.with_name(full.name + ".docgraph-rename.tmp")
            tmp.write_bytes(newline.join(lines).encode("utf-8", errors="surrogateescape"))
            os.replace(tmp, full)
            written.append(logical)
    return {"applied": applied, "files_written": written, "skipped": skipped,
            "sources": list(sources)}
