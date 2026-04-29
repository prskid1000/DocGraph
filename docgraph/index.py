"""Parallel indexer pipeline.

Walker → ProcessPool[parse] → Embed batcher → Bulk writer (Kuzu UNWIND)

Two-pass edge resolution: first all entities are written and a global
symbol table (qname → id, name → [ids]) is built; then edges are matched
against the table and bulk-inserted.

Tier 4 differentiator edges:
  SIMILAR_TO        — vector top-K
  CO_CHANGED_WITH   — git log --name-only over last N commits
  TESTS             — heuristic name match for test functions
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)

from docgraph.config import Config, MAX_FILE_BYTES
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.parse import detect_language, parse_file, FileParse
from docgraph.rank import compute_pagerank, write_pagerank

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None] | None


# --- Walker ---------------------------------------------------------------


def walk_files(cfg: Config) -> list[Path]:
    """Yield all parseable files under repo_root, respecting ignores."""
    out: list[Path] = []
    root = cfg.repo_root
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored dirs in-place
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        dirnames[:] = [
            d for d in dirnames
            if not cfg.is_ignored(f"{rel_dir}/{d}/" if rel_dir != "." else f"{d}/")
        ]
        for fname in filenames:
            full = Path(dirpath) / fname
            rel = str(full.relative_to(root)).replace("\\", "/")
            if cfg.is_ignored(rel):
                continue
            if detect_language(full) is None:
                continue
            try:
                if full.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            out.append(full)
    return out


# --- Parse worker (runs in subprocess) ------------------------------------


def _parse_worker(args: tuple[str, str]) -> dict | None:
    """Top-level so it's picklable. Returns serialized FileParse dict."""
    file_path, repo_root = args
    try:
        fp = parse_file(Path(file_path), Path(repo_root))
        if fp is None:
            return None
        return {
            "file": fp.file,
            "language": fp.language,
            "lines": fp.lines,
            "entities": [asdict(e) for e in fp.entities],
            "edges": [asdict(e) for e in fp.edges],
        }
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{file_path}: {e}"}


# --- Cache ----------------------------------------------------------------


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    try:
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


def load_cache(cfg: Config) -> dict[str, str]:
    if not cfg.cache_path.exists():
        return {}
    try:
        return json.loads(cfg.cache_path.read_text())
    except Exception:
        return {}


def save_cache(cfg: Config, cache: dict[str, str]) -> None:
    cfg.cache_path.write_text(json.dumps(cache))


# --- Main pipeline --------------------------------------------------------


class Indexer:
    def __init__(self, cfg: Config, db: GraphDB, embedder: Embedder | None = None):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder or Embedder(cfg.embedding_model)
        self._next_id = 1

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def index_all(self, incremental: bool = True, progress_cb: ProgressCb = None) -> dict:
        t0 = time.perf_counter()
        files = walk_files(self.cfg)
        cache = load_cache(self.cfg) if incremental else {}

        # Filter to changed files
        changed: list[Path] = []
        new_cache: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_to(self.cfg.repo_root)).replace("\\", "/")
            h = _file_hash(f)
            new_cache[rel] = h
            if cache.get(rel) != h:
                changed.append(f)

        log.info(f"{len(files)} files total, {len(changed)} to (re)parse")

        if not changed and incremental:
            return {"files": len(files), "changed": 0, "elapsed": 0.0}

        # MVP: any change → full rebuild. Real per-file delta updates need
        # CASCADE deletes which Kuzu doesn't ergonomically support yet.
        # Trade-off: simpler + always correct vs. slower on small edits.
        self.db.wipe(self.cfg.db_path)
        self.db = GraphDB(self.cfg.db_path, self.embedder.dim)
        self.db.init_schema()
        # Re-parse all files now (we'd lost the prior parse), so set changed = files
        changed = files

        # 1. Parse in parallel
        parsed: list[FileParse] = []
        errors: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as prog:
            ptask = prog.add_task("Parsing", total=len(changed))
            args_iter = [(str(f), str(self.cfg.repo_root)) for f in changed]
            with ProcessPoolExecutor(max_workers=self.cfg.workers) as ex:
                for result in ex.map(_parse_worker, args_iter, chunksize=8):
                    prog.advance(ptask)
                    if result is None:
                        continue
                    if "_error" in result:
                        errors.append(result["_error"])
                        continue
                    fp = FileParse(
                        file=result["file"],
                        language=result["language"],
                        lines=result["lines"],
                        entities=[_entity_from_dict(e) for e in result["entities"]],
                        edges=[_edge_from_dict(e) for e in result["edges"]],
                    )
                    parsed.append(fp)
                    if progress_cb:
                        progress_cb("parse", len(parsed), len(changed))

        # 2. Build node rows + symbol table
        log.info(f"Parsed {len(parsed)} files, {sum(len(p.entities) for p in parsed)} entities")
        file_rows: list[dict] = []
        class_rows: list[dict] = []
        function_rows: list[dict] = []
        variable_rows: list[dict] = []
        module_rows: dict[str, dict] = {}  # name → row

        # qname → (label, id)
        qname_index: dict[str, tuple[str, int]] = {}
        # name → list[(label, id, file)]  for fuzzy resolution
        name_index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        # file path → File node id
        file_index: dict[str, int] = {}
        # qname → (file, line_start) for embedding text retrieval
        embed_targets: list[tuple[str, int, str]] = []  # (label, id, text)

        for fp in parsed:
            fid = self._new_id()
            file_index[fp.file] = fid
            file_rows.append({
                "id": fid,
                "path": fp.file,
                "language": fp.language,
                "lines": fp.lines,
                "hash": new_cache.get(fp.file, ""),
                "pagerank": 0.0,
            })
            for ent in fp.entities:
                eid = self._new_id()
                qname_index[ent.qname] = (
                    "Class" if ent.kind in ("class", "interface") else
                    "Function" if ent.kind in ("function", "method") else
                    "Variable",
                    eid,
                )
                name_index[ent.name].append(
                    (qname_index[ent.qname][0], eid, ent.file)
                )
                if ent.kind in ("class", "interface"):
                    class_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": ent.file,
                        "line_start": ent.line_start,
                        "line_end": ent.line_end,
                        "body": ent.body,
                        "kind": ent.kind,
                        "embedding": [0.0] * self.embedder.dim,
                        "pagerank": 0.0,
                    })
                    embed_targets.append(("Class", eid, f"{ent.name}\n{ent.body[:1500]}"))
                elif ent.kind in ("function", "method"):
                    is_test = (
                        ent.name.startswith("test_") or
                        ent.name.startswith("test") and ent.name[4:5].isupper() or
                        "/test" in ent.file or "/tests/" in ent.file or "_test." in ent.file
                    )
                    function_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": ent.file,
                        "line_start": ent.line_start,
                        "line_end": ent.line_end,
                        "body": ent.body,
                        "signature": ent.signature or ent.body.split("\n")[0][:200],
                        "is_method": ent.kind == "method",
                        "is_test": is_test,
                        "embedding": [0.0] * self.embedder.dim,
                        "pagerank": 0.0,
                    })
                    embed_targets.append(("Function", eid, f"{ent.name}\n{ent.body[:1500]}"))
                else:
                    variable_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": ent.file,
                        "line": ent.line_start,
                        "scope": ent.extra.get("scope", "module"),
                    })

        # 3. Embed all targets in batches
        if embed_targets:
            with Progress(
                SpinnerColumn(),
                TextColumn("Embedding"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as prog:
                etask = prog.add_task("Embedding", total=len(embed_targets))
                vectors = self.embedder.embed(
                    [t[2] for t in embed_targets],
                    batch_size=self.cfg.embed_batch_size,
                )
                prog.advance(etask, len(embed_targets))
            # Splice vectors back into rows
            vec_by_id: dict[tuple[str, int], list[float]] = {}
            for (label, eid, _), vec in zip(embed_targets, vectors):
                vec_by_id[(label, eid)] = vec
            for row in class_rows:
                row["embedding"] = vec_by_id.get(("Class", row["id"]), [0.0] * self.embedder.dim)
            for row in function_rows:
                row["embedding"] = vec_by_id.get(("Function", row["id"]), [0.0] * self.embedder.dim)

        # 4. Bulk write nodes
        log.info("Writing nodes to Kuzu...")
        self.db.insert_nodes("File", file_rows)
        self.db.insert_nodes("Class", class_rows)
        self.db.insert_nodes("Function", function_rows)
        self.db.insert_nodes("Variable", variable_rows)

        # 5. Resolve and write edges
        log.info("Resolving edges...")
        contains_edges: list[tuple[str, str, dict]] = []
        # File CONTAINS class/function/variable
        for fp in parsed:
            fid = file_index[fp.file]
            for ent in fp.entities:
                if ent.qname not in qname_index:
                    continue
                label, eid = qname_index[ent.qname]
                # Only top-level (non-method) belong directly to file
                if "::" not in ent.qname.replace(fp.file + "::", "", 1):
                    contains_edges.append(("File", label, {"from_id": fid, "to_id": eid}))

        # Class CONTAINS method/var (parent qname is in qname_index)
        for fp in parsed:
            for ent in fp.entities:
                if ent.qname not in qname_index:
                    continue
                parts = ent.qname.split("::")
                if len(parts) >= 3:
                    parent_q = "::".join(parts[:-1])
                    if parent_q in qname_index:
                        plabel, pid = qname_index[parent_q]
                        elabel, eid = qname_index[ent.qname]
                        if plabel == "Class":
                            contains_edges.append(("Class", elabel, {"from_id": pid, "to_id": eid}))

        # Group by (from_label, to_label)
        from collections import defaultdict as _dd
        contains_groups: dict[tuple[str, str], list[dict]] = _dd(list)
        for from_lbl, to_lbl, row in contains_edges:
            contains_groups[(from_lbl, to_lbl)].append(row)
        for (fl, tl), rows in contains_groups.items():
            self.db.insert_edges("CONTAINS", fl, tl, rows)

        # CALLS, INSTANTIATES, REFERENCES_
        calls_rows: list[dict] = []
        inst_rows: list[dict] = []
        inherits_rows: list[dict] = []
        decorated_func_rows: list[dict] = []
        decorated_class_rows: list[dict] = []
        imports_file_rows: list[dict] = []
        imports_module_rows: list[dict] = []

        # Build set of qnames per file for scoped lookup
        qnames_by_file: dict[str, list[str]] = defaultdict(list)
        for q in qname_index:
            try:
                f = q.split("::")[0]
                qnames_by_file[f].append(q)
            except IndexError:
                continue

        def resolve(name: str, src_file: str, prefer_kind: str | None = None) -> tuple[str, int] | None:
            """Resolve a target name. Prefer same-file definitions; then any."""
            cands = name_index.get(name, [])
            if not cands:
                return None
            # Same file first
            same_file = [c for c in cands if c[2] == src_file]
            pool = same_file or cands
            if prefer_kind:
                pref = [c for c in pool if c[0] == prefer_kind]
                if pref:
                    pool = pref
            label, eid, _ = pool[0]
            return (label, eid)

        for fp in parsed:
            for edge in fp.edges:
                if edge.kind == "CALLS":
                    if edge.src_qname is None or edge.src_qname not in qname_index:
                        continue
                    src_label, src_id = qname_index[edge.src_qname]
                    if src_label != "Function":
                        continue
                    target = resolve(edge.target_name, fp.file, prefer_kind="Function")
                    if target and target[0] == "Function":
                        calls_rows.append({"from_id": src_id, "to_id": target[1], "line": edge.line})
                elif edge.kind == "INSTANTIATES":
                    if edge.src_qname is None or edge.src_qname not in qname_index:
                        continue
                    src_label, src_id = qname_index[edge.src_qname]
                    if src_label != "Function":
                        continue
                    target = resolve(edge.target_name, fp.file, prefer_kind="Class")
                    if target and target[0] == "Class":
                        inst_rows.append({"from_id": src_id, "to_id": target[1], "line": edge.line})
                elif edge.kind == "INHERITS":
                    if edge.src_qname is None or edge.src_qname not in qname_index:
                        continue
                    src_label, src_id = qname_index[edge.src_qname]
                    if src_label != "Class":
                        continue
                    target = resolve(edge.target_name, fp.file, prefer_kind="Class")
                    if target and target[0] == "Class":
                        inherits_rows.append({"from_id": src_id, "to_id": target[1]})
                elif edge.kind == "DECORATED_BY":
                    if edge.src_qname is None or edge.src_qname not in qname_index:
                        continue
                    src_label, src_id = qname_index[edge.src_qname]
                    target = resolve(edge.target_name, fp.file, prefer_kind="Function")
                    if target and target[0] == "Function":
                        if src_label == "Function":
                            decorated_func_rows.append({"from_id": src_id, "to_id": target[1]})
                        elif src_label == "Class":
                            decorated_class_rows.append({"from_id": src_id, "to_id": target[1]})
                elif edge.kind == "IMPORTS":
                    src_fid = file_index[fp.file]
                    # Try matching another File by path prefix
                    target_path = edge.target_name.replace(".", "/")
                    matched_fid = None
                    for path, fid in file_index.items():
                        if path.startswith(target_path) or target_path in path:
                            matched_fid = fid
                            break
                    if matched_fid:
                        imports_file_rows.append({"from_id": src_fid, "to_id": matched_fid})
                    else:
                        # Module node
                        if edge.target_name not in module_rows:
                            mid = self._new_id()
                            module_rows[edge.target_name] = {
                                "id": mid,
                                "name": edge.target_name,
                                "language": fp.language,
                            }
                        mid = module_rows[edge.target_name]["id"]
                        imports_module_rows.append({"from_id": src_fid, "to_id": mid})

        if module_rows:
            self.db.insert_nodes("Module", list(module_rows.values()))

        self.db.insert_edges("CALLS", "Function", "Function", calls_rows)
        self.db.insert_edges("INSTANTIATES", "Function", "Class", inst_rows)
        self.db.insert_edges("INHERITS", "Class", "Class", inherits_rows)
        self.db.insert_edges("DECORATED_BY", "Function", "Function", decorated_func_rows)
        self.db.insert_edges("DECORATED_BY", "Class", "Function", decorated_class_rows)
        self.db.insert_edges("IMPORTS", "File", "File", imports_file_rows)
        self.db.insert_edges("IMPORTS", "File", "Module", imports_module_rows)

        # 6. Tier 4 — SIMILAR_TO
        log.info("Computing SIMILAR_TO edges...")
        self._write_similar_edges(function_rows, "Function")
        self._write_similar_edges(class_rows, "Class")

        # 7. Tier 4 — CO_CHANGED_WITH
        log.info("Computing CO_CHANGED_WITH from git history...")
        self._write_co_changed(file_index)

        # 8. Tier 4 — TESTS
        log.info("Linking TESTS edges...")
        self._write_tests_edges(function_rows, name_index, qname_index)

        # 9. PageRank
        log.info("Running PageRank...")
        scores = compute_pagerank(self.db)
        write_pagerank(self.db, scores)

        # 10. Save cache
        save_cache(self.cfg, new_cache)

        elapsed = time.perf_counter() - t0
        return {
            "files": len(files),
            "changed": len(changed),
            "entities": sum(len(p.entities) for p in parsed),
            "elapsed": elapsed,
            "errors": len(errors),
        }

    def _write_similar_edges(self, rows: list[dict], label: str) -> None:
        if len(rows) < 2:
            return
        import numpy as np
        ids = [r["id"] for r in rows]
        mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        # Cosine similarity matrix (chunked for memory if huge)
        sim_edges: list[dict] = []
        k = self.cfg.similar_top_k
        n = len(ids)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = mat[start:end] @ mat.T  # (chunk, n)
            for i in range(end - start):
                row = sims[i]
                row[start + i] = -1  # exclude self
                top = np.argpartition(-row, k)[:k]
                for j in top:
                    score = float(row[j])
                    if score < 0.5:
                        continue
                    sim_edges.append({
                        "from_id": ids[start + i],
                        "to_id": ids[int(j)],
                        "score": score,
                    })
        if sim_edges:
            self.db.insert_edges("SIMILAR_TO", label, label, sim_edges)

    def _write_co_changed(self, file_index: dict[str, int]) -> None:
        try:
            out = subprocess.check_output(
                ["git", "log", f"-{self.cfg.co_change_window}", "--name-only", "--pretty=format:---"],
                cwd=self.cfg.repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        commits: list[set[str]] = []
        cur: set[str] = set()
        for line in out.splitlines():
            if line.startswith("---"):
                if cur:
                    commits.append(cur)
                cur = set()
            elif line.strip():
                cur.add(line.strip().replace("\\", "/"))
        if cur:
            commits.append(cur)

        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        for commit_files in commits:
            files = [f for f in commit_files if f in file_index]
            for i, a in enumerate(files):
                for b in files[i + 1:]:
                    pair = tuple(sorted([a, b]))  # type: ignore
                    pair_count[pair] += 1

        rows = [
            {"from_id": file_index[a], "to_id": file_index[b], "count": c}
            for (a, b), c in pair_count.items() if c >= 2
        ]
        if rows:
            self.db.insert_edges("CO_CHANGED_WITH", "File", "File", rows)

    def _write_tests_edges(
        self,
        function_rows: list[dict],
        name_index: dict,
        qname_index: dict,
    ) -> None:
        rows_func: list[dict] = []
        rows_class: list[dict] = []
        for fr in function_rows:
            if not fr["is_test"]:
                continue
            stripped = fr["name"]
            for prefix in ("test_", "test"):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip("_")
                    break
            if not stripped:
                continue
            cands = name_index.get(stripped, [])
            for label, eid, _ in cands:
                if eid == fr["id"]:
                    continue
                if label == "Function":
                    rows_func.append({"from_id": fr["id"], "to_id": eid})
                elif label == "Class":
                    rows_class.append({"from_id": fr["id"], "to_id": eid})
        if rows_func:
            self.db.insert_edges("TESTS", "Function", "Function", rows_func)
        if rows_class:
            self.db.insert_edges("TESTS", "Function", "Class", rows_class)


def _entity_from_dict(d: dict):
    from docgraph.parse import Entity
    return Entity(**d)


def _edge_from_dict(d: dict):
    from docgraph.parse import RawEdge
    return RawEdge(**d)
