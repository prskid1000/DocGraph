"""Parallel indexer pipeline with per-file delta updates.

Cache (.docgraph/cache.json) stores per-file `{hash, entities, edges}` so
incremental runs can:
  1. DETACH DELETE only changed files' nodes (incident edges removed too).
  2. Re-parse only changed files.
  3. Re-resolve edges that crossed the changed/unchanged boundary
     (unchanged-file edges into unchanged-file targets are still in the DB
     and don't need to be touched).

Tier 4 differentiator edges (SIMILAR_TO, CO_CHANGED_WITH, TESTS) are
always recomputed because they're cheap and global.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)

from docgraph.config import Config, MAX_FILE_BYTES
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.parse import detect_language, parse_file, FileParse, Entity, RawEdge
from docgraph.rank import compute_pagerank, write_pagerank
from docgraph.summary import build_embedding_text, chunk_body

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None] | None


# --- Walker ---------------------------------------------------------------


def walk_files(cfg: Config) -> list[tuple[Path, str]]:
    """Return [(absolute_path, logical_rel)]. logical_rel includes a `<repo>/`
    prefix in multi-root mode; in single-root mode it's just the rel path."""
    out: list[tuple[Path, str]] = []
    for root, prefix in cfg.roots_with_prefix():
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
            dirnames[:] = [
                d for d in dirnames
                if not cfg.is_ignored(
                    f"{rel_dir}/{d}/" if rel_dir != "." else f"{d}/", root=root
                )
            ]
            for fname in filenames:
                full = Path(dirpath) / fname
                rel = str(full.relative_to(root)).replace("\\", "/")
                if cfg.is_ignored(rel, root=root):
                    continue
                if detect_language(full) is None:
                    continue
                try:
                    if full.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                out.append((full, f"{prefix}{rel}"))
    return out


# --- Parse worker ---------------------------------------------------------


def _parse_worker(args: tuple[str, str, str]) -> dict | None:
    file_path, repo_root, rel_override = args
    try:
        fp = parse_file(Path(file_path), Path(repo_root), rel_override=rel_override)
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


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    try:
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


# --- Cache ----------------------------------------------------------------


def load_cache(cfg: Config) -> dict[str, dict]:
    if not cfg.cache_path.exists():
        return {}
    try:
        return json.loads(cfg.cache_path.read_text())
    except Exception:
        return {}


def save_cache(cfg: Config, cache: dict[str, dict]) -> None:
    cfg.cache_path.write_text(json.dumps(cache))


# --- Indexer --------------------------------------------------------------


class Indexer:
    def __init__(self, cfg: Config, db: GraphDB, embedder: Embedder | None = None):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder or Embedder(cfg.embedding_model)
        self._next_id = 1

    # ---- ID allocation ----
    def _seed_ids_from_db(self) -> None:
        """Continue allocating after the max id currently in the DB."""
        max_id = 0
        for label in ("File", "Module", "Class", "Function", "Variable", "Chunk"):
            try:
                rows = self.db.fetch_all(f"MATCH (n:{label}) RETURN max(n.id) AS m")
                m = rows[0]["m"] if rows and rows[0]["m"] is not None else 0
                if m > max_id:
                    max_id = m
            except Exception:
                pass
        self._next_id = max_id + 1

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # ---- DB delete ----
    def _delete_files_from_db(self, files: list[str]) -> None:
        """DETACH DELETE all entities + the File node for each path."""
        if not files:
            return
        # Delete entities first (matches all by .file property)
        for label in ("Function", "Class", "Variable"):
            self.db.execute(
                f"MATCH (n:{label}) WHERE n.file IN $files DETACH DELETE n",
                {"files": files},
            )
        # Sub-function chunks (separate node table; not auto-cascaded)
        try:
            self.db.execute(
                "MATCH (n:Chunk) WHERE n.file IN $files DETACH DELETE n",
                {"files": files},
            )
        except Exception:
            pass
        # Then delete File nodes
        self.db.execute(
            "MATCH (n:File) WHERE n.path IN $files DETACH DELETE n",
            {"files": files},
        )
        # Tier 4: delete CO_CHANGED edges involving these (will be recomputed)
        # SIMILAR_TO already gone via DETACH DELETE on Function/Class
        # Module nodes left alone (cheap, may be reused; orphans tolerated)

    # ---- Main entrypoint ----
    def index_all(self, incremental: bool = True, progress_cb: ProgressCb = None) -> dict:
        t0 = time.perf_counter()
        cache = load_cache(self.cfg) if incremental else {}
        files_on_disk = walk_files(self.cfg)
        # logical_rel → absolute path
        on_disk_rel: dict[str, Path] = {rel: path for path, rel in files_on_disk}

        # Compute hashes; identify changed/added/deleted
        changed: list[tuple[Path, str]] = []  # (absolute_path, logical_rel)
        unchanged_rels: set[str] = set()
        new_hashes: dict[str, str] = {}
        for rel, path in on_disk_rel.items():
            h = _file_hash(path)
            new_hashes[rel] = h
            cached = cache.get(rel)
            if cached and cached.get("hash") == h:
                unchanged_rels.add(rel)
            else:
                changed.append((path, rel))

        deleted_rels = [rel for rel in cache.keys() if rel not in on_disk_rel]
        log.info(
            f"{len(on_disk_rel)} files: {len(changed)} changed/added, "
            f"{len(deleted_rels)} deleted, {len(unchanged_rels)} unchanged"
        )

        # No changes: bail
        if incremental and not changed and not deleted_rels:
            return {
                "files": len(on_disk_rel), "changed": 0, "deleted": 0,
                "entities": sum(len(c.get("entities", [])) for c in cache.values()),
                "elapsed": time.perf_counter() - t0, "errors": 0,
            }

        # Full reindex path
        if not incremental:
            self.db.wipe(self.cfg.db_path)
            self.db = GraphDB(self.cfg.db_path, self.embedder.dim)
            self.db.init_schema()
            self._next_id = 1
            cache = {}
            unchanged_rels = set()
            deleted_rels = []
            changed = list(files_on_disk)

        # ---- Step 1: delete affected nodes from DB ----
        affected = [rel for _path, rel in changed]
        self._delete_files_from_db(affected + deleted_rels)
        for rel in deleted_rels:
            cache.pop(rel, None)

        # ---- Step 2: parse changed files in parallel ----
        parsed: dict[str, FileParse] = {}
        errors: list[str] = []
        if changed:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as prog:
                ptask = prog.add_task("Parsing", total=len(changed))
                # parse worker receives (absolute_path, owning_root, logical_rel)
                args_iter = []
                roots = self.cfg.roots_with_prefix()
                for path, logical_rel in changed:
                    owner = self.cfg.repo_root
                    for root, prefix in roots:
                        if prefix == "" or logical_rel.startswith(prefix):
                            owner = root
                            break
                    args_iter.append((str(path), str(owner), logical_rel))
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
                            entities=[Entity(**e) for e in result["entities"]],
                            edges=[RawEdge(**e) for e in result["edges"]],
                        )
                        parsed[fp.file] = fp
                        # Update cache
                        cache[fp.file] = {
                            "hash": new_hashes[fp.file],
                            "language": fp.language,
                            "lines": fp.lines,
                            "entities": result["entities"],
                            "edges": result["edges"],
                        }

        # ---- Step 3: seed ID allocator ----
        self._seed_ids_from_db()

        # ---- Step 4: build node rows for newly-parsed files ----
        file_rows: list[dict] = []
        class_rows: list[dict] = []
        function_rows: list[dict] = []
        variable_rows: list[dict] = []
        # qname → (label, id) for ALL entities (across cache, for edge resolution)
        # plus tracking which are new for embedding
        new_embed_targets: list[tuple[str, int, str]] = []  # (label, id, text) for embedding

        for rel, fp in parsed.items():
            fid = self._new_id()
            file_rows.append({
                "id": fid,
                "path": rel,
                "language": fp.language,
                "lines": fp.lines,
                "hash": new_hashes[rel],
                "pagerank": 0.0,
            })
            for ent in fp.entities:
                eid = self._new_id()
                # Stash id back into cache entity for lookups later
                # (we'll rebuild the cache entity with ids below)
                ent.extra["_id"] = eid
                if ent.kind in ("class", "interface"):
                    class_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": rel,
                        "line_start": ent.line_start,
                        "line_end": ent.line_end,
                        "body": ent.body,
                        "kind": ent.kind,
                        "embedding": [0.0] * self.embedder.dim,
                        "pagerank": 0.0,
                    })
                    new_embed_targets.append((
                        "Class",
                        eid,
                        build_embedding_text(
                            ent.name, ent.qname, ent.signature, ent.body,
                            fp.language, ent.kind,
                        ),
                    ))
                elif ent.kind in ("function", "method"):
                    is_test = (
                        ent.name.startswith("test_") or
                        (ent.name.startswith("test") and len(ent.name) > 4 and ent.name[4:5].isupper()) or
                        "/test" in rel or "/tests/" in rel or "_test." in rel
                    )
                    function_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": rel,
                        "line_start": ent.line_start,
                        "line_end": ent.line_end,
                        "body": ent.body,
                        "signature": ent.signature or ent.body.split("\n")[0][:200],
                        "is_method": ent.kind == "method",
                        "is_test": is_test,
                        "embedding": [0.0] * self.embedder.dim,
                        "pagerank": 0.0,
                    })
                    new_embed_targets.append((
                        "Function",
                        eid,
                        build_embedding_text(
                            ent.name, ent.qname, ent.signature, ent.body,
                            fp.language, ent.kind,
                        ),
                    ))
                else:
                    variable_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": rel,
                        "line": ent.line_start,
                        "scope": ent.extra.get("scope", "module") if isinstance(ent.extra, dict) else "module",
                    })

        # ---- Step 5: embed new entities ----
        if new_embed_targets:
            with Progress(
                SpinnerColumn(),
                TextColumn("Embedding"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as prog:
                etask = prog.add_task("Embedding", total=len(new_embed_targets))
                vectors = self.embedder.embed(
                    [t[2] for t in new_embed_targets],
                    batch_size=self.cfg.embed_batch_size,
                )
                prog.advance(etask, len(new_embed_targets))
            vec_by_id: dict[tuple[str, int], list[float]] = {}
            for (label, eid, _), vec in zip(new_embed_targets, vectors):
                vec_by_id[(label, eid)] = vec
            for r in class_rows:
                r["embedding"] = vec_by_id.get(("Class", r["id"]), [0.0] * self.embedder.dim)
            for r in function_rows:
                r["embedding"] = vec_by_id.get(("Function", r["id"]), [0.0] * self.embedder.dim)

        # ---- Step 5b: build sub-chunks for long entities ----
        chunk_rows: list[dict] = []
        chunk_contains_func: list[dict] = []
        chunk_contains_class: list[dict] = []
        # Walk parsed entities + use the freshly-assigned ids stored on ent.extra
        for rel, fp in parsed.items():
            for ent in fp.entities:
                eid = ent.extra.get("_id") if isinstance(ent.extra, dict) else None
                if eid is None:
                    continue
                if ent.kind not in ("function", "method", "class", "interface"):
                    continue
                pieces = chunk_body(ent.body or "")
                if not pieces:
                    continue
                parent_label = "Class" if ent.kind in ("class", "interface") else "Function"
                for idx, piece in enumerate(pieces):
                    cid = self._new_id()
                    chunk_rows.append({
                        "id": cid,
                        "parent_qname": ent.qname,
                        "parent_label": parent_label,
                        "file": rel,
                        "idx": idx,
                        "body": piece[:6000],
                        "embedding": [0.0] * self.embedder.dim,
                    })
                    if parent_label == "Function":
                        chunk_contains_func.append({"from_id": eid, "to_id": cid})
                    else:
                        chunk_contains_class.append({"from_id": eid, "to_id": cid})

        # Embed all chunks in one batch
        if chunk_rows:
            chunk_texts = [r["body"] for r in chunk_rows]
            chunk_vecs = self.embedder.embed(chunk_texts, batch_size=self.cfg.embed_batch_size)
            for r, v in zip(chunk_rows, chunk_vecs):
                r["embedding"] = v

        # ---- Step 6: write new nodes ----
        log.info(
            f"Writing {len(file_rows)} files, {len(class_rows)} classes, "
            f"{len(function_rows)} functions, {len(variable_rows)} variables, "
            f"{len(chunk_rows)} chunks"
        )
        self.db.insert_nodes("File", file_rows)
        self.db.insert_nodes("Class", class_rows)
        self.db.insert_nodes("Function", function_rows)
        self.db.insert_nodes("Variable", variable_rows)
        self.db.insert_nodes("Chunk", chunk_rows)
        self.db.insert_edges("CONTAINS_CHUNK", "Function", "Chunk", chunk_contains_func)
        self.db.insert_edges("CONTAINS_CHUNK", "Class", "Chunk", chunk_contains_class)

        # ---- Step 7: build full symbol table from DB ----
        # qname → (label, id); name → list[(label, id, file)]
        qname_index: dict[str, tuple[str, int]] = {}
        name_index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        file_index: dict[str, int] = {}
        for label in ("Function", "Class", "Variable"):
            for r in self.db.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, "
                f"n.qname AS qname, n.file AS file"
            ):
                qname_index[r["qname"]] = (label, r["id"])
                name_index[r["name"]].append((label, r["id"], r["file"]))
        for r in self.db.fetch_all("MATCH (f:File) RETURN f.id AS id, f.path AS path"):
            file_index[r["path"]] = r["id"]

        changed_set = set(parsed.keys())  # for "edge needs reinsertion?" check

        # Scope-aware resolution: per-file set of files that file imports.
        # Built from the same fuzzy IMPORTS-target match we use for the edge
        # itself, so resolve() can prefer call-targets in imported files over
        # same-named symbols in unrelated files. This is the best we can do
        # without a real LSP — in practice it kills most cross-file CALLS
        # hallucinations on overloads / generics / re-exports.
        file_imports: dict[str, set[str]] = defaultdict(set)
        for rel, file_data in cache.items():
            for raw in file_data.get("edges", []):
                if raw.get("kind") != "IMPORTS":
                    continue
                target_name = raw.get("target_name") or ""
                target_path = target_name.replace(".", "/")
                if not target_path:
                    continue
                for path in file_index:
                    if path == rel:
                        continue
                    if path.startswith(target_path) or target_path in path:
                        file_imports[rel].add(path)
                        break

        def needs_insert(src_file: str, target_file: str | None) -> bool:
            """True if either endpoint was just (re)created."""
            if src_file in changed_set:
                return True
            if target_file and target_file in changed_set:
                return True
            return False

        def resolve(name: str, src_file: str, prefer_kind: str | None = None) -> tuple[str, int, str] | None:
            """Resolve a name. Prefer same-file → imported-file → global.
            Returns (label, id, file)."""
            cands = name_index.get(name, [])
            if not cands:
                return None
            same_file = [c for c in cands if c[2] == src_file]
            if same_file:
                pool = same_file
            else:
                imported = file_imports.get(src_file, ())
                from_imports = [c for c in cands if c[2] in imported]
                pool = from_imports or cands
            if prefer_kind:
                pref = [c for c in pool if c[0] == prefer_kind]
                if pref:
                    pool = pref
            return pool[0]

        # ---- Step 8: re-resolve and write edges ----
        # Combine RawEdges from cache (covers both unchanged and just-parsed files).
        # For each edge, decide if it needs DB insertion.
        contains_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        calls_rows: list[dict] = []
        inst_rows: list[dict] = []
        inherits_rows: list[dict] = []
        decorated_func_rows: list[dict] = []
        decorated_class_rows: list[dict] = []
        imports_file_rows: list[dict] = []
        imports_module_rows: list[dict] = []
        module_rows_by_name: dict[str, dict] = {}

        # Pre-load existing modules so we don't duplicate
        for r in self.db.fetch_all("MATCH (m:Module) RETURN m.id AS id, m.name AS name"):
            module_rows_by_name[r["name"]] = {"id": r["id"], "name": r["name"], "language": ""}

        # CONTAINS edges from cached entities (only for changed files; unchanged are still in DB)
        for rel, file_data in cache.items():
            if rel not in changed_set:
                continue
            fid = file_index.get(rel)
            if fid is None:
                continue
            # qnames in this file
            for ent_dict in file_data["entities"]:
                qname = ent_dict["qname"]
                if qname not in qname_index:
                    continue
                label, eid = qname_index[qname]
                # Only top-level entities are contained directly in File
                # Class methods are CONTAINS'd by Class (handled below)
                parts = qname.split("::")
                if len(parts) == 2:
                    contains_groups[("File", label)].append({"from_id": fid, "to_id": eid})
                elif len(parts) >= 3:
                    parent_q = "::".join(parts[:-1])
                    if parent_q in qname_index:
                        plabel, pid = qname_index[parent_q]
                        if plabel == "Class":
                            contains_groups[("Class", label)].append({"from_id": pid, "to_id": eid})

        # Other edges from cached RawEdges
        for rel, file_data in cache.items():
            for raw in file_data.get("edges", []):
                kind = raw["kind"]
                src_qname = raw.get("src_qname")
                target_name = raw.get("target_name")
                src_file = rel
                line = raw.get("line", 0)

                if kind == "IMPORTS":
                    if not needs_insert(src_file, None):
                        # Try to find the resolved file target to also check
                        pass
                    src_fid = file_index.get(src_file)
                    if src_fid is None:
                        continue
                    target_path = (target_name or "").replace(".", "/")
                    matched_fid = None
                    matched_path = None
                    for path, fid_ in file_index.items():
                        if path.startswith(target_path) or target_path in path:
                            matched_fid = fid_
                            matched_path = path
                            break
                    if matched_fid:
                        if needs_insert(src_file, matched_path):
                            imports_file_rows.append({"from_id": src_fid, "to_id": matched_fid})
                    else:
                        if not needs_insert(src_file, None):
                            continue
                        if target_name not in module_rows_by_name:
                            mid = self._new_id()
                            module_rows_by_name[target_name] = {
                                "id": mid, "name": target_name,
                                "language": file_data.get("language", ""),
                            }
                        mid = module_rows_by_name[target_name]["id"]
                        imports_module_rows.append({"from_id": src_fid, "to_id": mid})
                    continue

                if not src_qname or src_qname not in qname_index:
                    continue
                src_label, src_id = qname_index[src_qname]

                if kind == "CALLS":
                    if src_label != "Function":
                        continue
                    target = resolve(target_name, src_file, prefer_kind="Function")
                    if target and target[0] == "Function":
                        if needs_insert(src_file, target[2]):
                            calls_rows.append({"from_id": src_id, "to_id": target[1], "line": line})
                elif kind == "INSTANTIATES":
                    if src_label != "Function":
                        continue
                    target = resolve(target_name, src_file, prefer_kind="Class")
                    if target and target[0] == "Class":
                        if needs_insert(src_file, target[2]):
                            inst_rows.append({"from_id": src_id, "to_id": target[1], "line": line})
                elif kind == "INHERITS":
                    if src_label != "Class":
                        continue
                    target = resolve(target_name, src_file, prefer_kind="Class")
                    if target and target[0] == "Class":
                        if needs_insert(src_file, target[2]):
                            inherits_rows.append({"from_id": src_id, "to_id": target[1]})
                elif kind == "DECORATED_BY":
                    target = resolve(target_name, src_file, prefer_kind="Function")
                    if target and target[0] == "Function":
                        if needs_insert(src_file, target[2]):
                            if src_label == "Function":
                                decorated_func_rows.append({"from_id": src_id, "to_id": target[1]})
                            elif src_label == "Class":
                                decorated_class_rows.append({"from_id": src_id, "to_id": target[1]})

        # Insert new modules
        new_modules = [
            v for k, v in module_rows_by_name.items()
            if not self.db.fetch_all(
                "MATCH (m:Module) WHERE m.id = $id RETURN m.id", {"id": v["id"]}
            )
        ]
        if new_modules:
            self.db.insert_nodes("Module", new_modules)

        # Insert collected edges
        for (fl, tl), rows in contains_groups.items():
            self.db.insert_edges("CONTAINS", fl, tl, rows)
        self.db.insert_edges("CALLS", "Function", "Function", calls_rows)
        self.db.insert_edges("INSTANTIATES", "Function", "Class", inst_rows)
        self.db.insert_edges("INHERITS", "Class", "Class", inherits_rows)
        self.db.insert_edges("DECORATED_BY", "Function", "Function", decorated_func_rows)
        self.db.insert_edges("DECORATED_BY", "Class", "Function", decorated_class_rows)
        self.db.insert_edges("IMPORTS", "File", "File", imports_file_rows)
        self.db.insert_edges("IMPORTS", "File", "Module", imports_module_rows)

        # ---- Step 9: Tier 4 + PageRank (always recomputed; cheap and global) ----
        # Wipe and rebuild SIMILAR_TO and CO_CHANGED_WITH; PageRank too.
        try:
            self.db.execute("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
            self.db.execute("MATCH ()-[r:CO_CHANGED_WITH]->() DELETE r")
            self.db.execute("MATCH ()-[r:TESTS]->() DELETE r")
        except Exception:
            pass

        # Re-pull current Function/Class rows for similarity
        log.info("Computing SIMILAR_TO edges...")
        sim_rows = self.db.fetch_all(
            "MATCH (n:Function) RETURN n.id AS id, n.embedding AS embedding"
        )
        self._write_similar_edges(sim_rows, "Function")
        sim_rows = self.db.fetch_all(
            "MATCH (n:Class) RETURN n.id AS id, n.embedding AS embedding"
        )
        self._write_similar_edges(sim_rows, "Class")

        log.info("Computing CO_CHANGED_WITH from git history...")
        # Refresh file_index post-insert
        file_index = {
            r["path"]: r["id"]
            for r in self.db.fetch_all("MATCH (f:File) RETURN f.id AS id, f.path AS path")
        }
        self._write_co_changed(file_index)

        log.info("Linking TESTS edges...")
        function_rows_db = self.db.fetch_all(
            "MATCH (n:Function) RETURN n.id AS id, n.name AS name, n.is_test AS is_test"
        )
        # Build quick name index from DB
        name_index2: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for r in self.db.fetch_all("MATCH (n:Function) RETURN n.id AS id, n.name AS name"):
            name_index2[r["name"]].append(("Function", r["id"]))
        for r in self.db.fetch_all("MATCH (n:Class) RETURN n.id AS id, n.name AS name"):
            name_index2[r["name"]].append(("Class", r["id"]))
        self._write_tests_edges(function_rows_db, name_index2)

        log.info("Running PageRank...")
        scores = compute_pagerank(self.db)
        write_pagerank(self.db, scores)

        # ---- Step 10: persist cache (strip embeddings/IDs from entity dicts) ----
        # Cache entities should not carry _id (transient); strip.
        for rel in cache:
            for ent in cache[rel].get("entities", []):
                if "extra" in ent and isinstance(ent["extra"], dict):
                    ent["extra"].pop("_id", None)
        save_cache(self.cfg, cache)

        elapsed = time.perf_counter() - t0
        return {
            "files": len(on_disk_rel),
            "changed": len(parsed),
            "deleted": len(deleted_rels),
            "entities": sum(len(c.get("entities", [])) for c in cache.values()),
            "elapsed": elapsed,
            "errors": len(errors),
        }

    # ---- Tier 4 helpers ----
    def _write_similar_edges(self, rows: list[dict], label: str) -> None:
        if len(rows) < 2:
            return
        import numpy as np
        ids = [r["id"] for r in rows]
        try:
            mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
        except Exception:
            return
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        sim_edges: list[dict] = []
        k = self.cfg.similar_top_k
        n = len(ids)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = mat[start:end] @ mat.T
            for i in range(end - start):
                row = sims[i]
                row[start + i] = -1
                top_k = min(k, n - 1)
                if top_k <= 0:
                    continue
                top = np.argpartition(-row, top_k)[:top_k]
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
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        for root, prefix in self.cfg.roots_with_prefix():
            try:
                out = subprocess.check_output(
                    ["git", "log", f"-{self.cfg.co_change_window}", "--name-only", "--pretty=format:---"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
            commits: list[set[str]] = []
            cur: set[str] = set()
            for line in out.splitlines():
                if line.startswith("---"):
                    if cur:
                        commits.append(cur)
                    cur = set()
                elif line.strip():
                    cur.add(prefix + line.strip().replace("\\", "/"))
            if cur:
                commits.append(cur)

            for commit_files in commits:
                files = [f for f in commit_files if f in file_index]
                for i, a in enumerate(files):
                    for b in files[i + 1:]:
                        pair = tuple(sorted([a, b]))
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
        name_index: dict[str, list[tuple[str, int]]],
    ) -> None:
        rows_func: list[dict] = []
        rows_class: list[dict] = []
        for fr in function_rows:
            if not fr.get("is_test"):
                continue
            stripped = fr["name"]
            for prefix in ("test_", "test"):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip("_")
                    break
            if not stripped:
                continue
            for label, eid in name_index.get(stripped, []):
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
