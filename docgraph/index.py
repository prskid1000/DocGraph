"""Parallel indexer pipeline with per-file delta updates.

The per-file cache (`.docgraph/cache/*.json`, sharded -- see live.py) stores
`{hash, size, mtime, entities, edges}` so an incremental run:
  1. scans with cached decisions (or takes the watcher's paths), stats first;
  2. DETACH DELETEs only changed files' nodes (by id, from the live symbol
     table) and re-parses only changed + added files;
  3. re-resolves only the edges that can reach the change (_resolve_scope);
  4. patches Tier 4 edges, ranks, positions, communities and the tile
     sidecar for the new ids only -- the global passes run on a full index
     or, past the drift limit, as background maintenance on the host.
Every stage is timed (`timings` in the result, `timings_ms` in state.json).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from docgraph.cancel import CancelToken
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

_console = Console()


def _bar() -> Progress:
    """ML-training-style progress bar: spinner + desc + bar + % + M/N + elapsed + ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
    )

from docgraph.config import Config, MAX_FILE_BYTES
from docgraph.db import EdgeJournal, GraphDB, SCHEMA_VERSION
from docgraph.embed import Embedder, resolve_device
from docgraph.parse import classify_file, FileParse, Entity, RawEdge
from docgraph.resolve import SymbolResolver, candidate_confidence
from docgraph.summary import build_embedding_text, chunk_body
from docgraph.proc_util import NO_WINDOW
from docgraph import history as _history
from docgraph.kwindex import KeywordBuilder
from docgraph.live import LiveIndex, classify_paths, scan as live_scan, test_target

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None] | None

# Bounded list of removed symbols kept in state.json for symbol_history.
MAX_REMOVED_SYMBOLS = 500


def _ehash(model: str, text: str) -> str:
    """Content hash that keys the embedding cache: same text under the same
    model -> same vector, wherever the entity moved to."""
    h = hashlib.sha1()
    h.update(model.encode("utf-8", "replace"))
    h.update(b"\0")
    h.update(text.encode("utf-8", "replace"))
    return h.hexdigest()


def _wire_extra_paths(cfg: Config) -> None:
    """Re-read repos.json at index time and patch cfg.extra_roots for any
    paths added after the host started (e.g. via /api/repos or the tray UI).
    Safe per the workspace writer lock that serializes index/wiki runs."""
    import json as _json
    repos_file = cfg.data_dir / "repos.json"
    if not repos_file.exists():
        return
    try:
        raw = _json.loads(repos_file.read_text(encoding="utf-8"))
        paths = [Path(p).resolve() for p in raw if p]
    except Exception:
        return

    for p in paths:
        if p == cfg.repo_root or p in cfg.extra_roots:
            continue
        if not p.exists():
            log.debug("_wire_extra_paths: skipping missing path %s", p)
            continue
        cfg.extra_roots.append(p)
        try:
            from docgraph.ignores import assemble_ignores
            import pathspec as _ps
            index_patterns, ecosystems = assemble_ignores(p)
            # Mirror Config.__post_init__: read user-level ignore files so
            # .gitignore / .docgraphignore inside the extra path are honoured.
            user_patterns: list[str] = []
            for _fname in (".gitignore", ".docgraphignore", ".cursorindexingignore"):
                _ign = p / _fname
                if _ign.exists():
                    try:
                        user_patterns.extend(
                            _ign.read_text(encoding="utf-8", errors="ignore").splitlines()
                        )
                    except Exception:
                        pass
            index_patterns.extend(user_patterns)
            cfg.ignore_specs[p] = _ps.PathSpec.from_lines("gitignore", index_patterns)
            cfg.user_ignore_specs[p] = _ps.PathSpec.from_lines("gitignore", user_patterns)
            ai_patterns: list[str] = []
            _ci = p / ".cursorignore"
            if _ci.exists():
                try:
                    ai_patterns.extend(
                        _ci.read_text(encoding="utf-8", errors="ignore").splitlines()
                    )
                except Exception:
                    pass
            cfg.ai_block_specs[p] = _ps.PathSpec.from_lines("gitignore", ai_patterns)
            cfg.detected_ecosystems[p] = ecosystems
        except Exception as exc:
            log.warning("_wire_extra_paths: ignore-spec setup failed for %s: %s", p, exc)


def _maybe_fetch_links(
    cfg: Config,
    force: bool = False,
    cancel_check: "Callable[[], None] | None" = None,
    progress_cb: "Callable[[int, int, int], None] | None" = None,
) -> None:
    """Fetch stale external links and wire external_dir into cfg.extra_roots.

    Called before index_all() (and build_wiki()). If links.json has entries,
    runs the fetch step then appends cfg.external_dir to cfg.extra_roots so
    walk_files() picks up the downloaded HTML pages. Patches ignore_specs in
    place — safe because index/wiki runs are serialized per root via the
    workspace writer lock.
    """
    try:
        from docgraph.links import load_links
        from docgraph.fetch import fetch_all
    except ImportError:
        return

    if not load_links(cfg.data_dir):
        return

    external_dir = cfg.external_dir
    # If pages were wiped (e.g. after `docgraph clear`) the TTL timestamp in
    # links.json is stale relative to the file-system state — the link looks
    # fresh but has no cached pages. Force a re-fetch in that case so a Clear
    # + Index cycle doesn't silently produce 0 entities.
    pages_missing = not external_dir.exists() or not list(external_dir.glob("*.html"))
    fetch_all(cfg.data_dir, force=force or pages_missing, cancel_check=cancel_check,
              progress_cb=progress_cb)

    if not external_dir.exists() or not list(external_dir.glob("*.html")):
        return

    if external_dir in cfg.extra_roots:
        return

    cfg.extra_roots.append(external_dir)
    try:
        from docgraph.ignores import assemble_ignores
        import pathspec as _ps
        patterns, _ = assemble_ignores(external_dir)
        cfg.ignore_specs[external_dir] = _ps.PathSpec.from_lines("gitignore", patterns)
        cfg.user_ignore_specs[external_dir] = _ps.PathSpec.from_lines("gitignore", [])
        cfg.ai_block_specs[external_dir] = _ps.PathSpec.from_lines("gitignore", [])
        cfg.detected_ecosystems[external_dir] = []
    except Exception as exc:
        log.warning("_maybe_fetch_links: ignore-spec setup failed: %s", exc)


# --- Walker ---------------------------------------------------------------


def walk_files(cfg: Config) -> list[tuple[Path, str]]:
    """Return [(absolute_path, logical_rel)]. logical_rel includes a `<repo>/`
    prefix in multi-root mode; in single-root mode it's just the rel path."""
    out: list[tuple[Path, str]] = []
    text_fallback = bool(getattr(cfg, "text_fallback", True))
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
                try:
                    if full.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                kind = classify_file(full, sniff_known=False)
                if kind is None:
                    continue
                if kind.startswith("text:") and not text_fallback:
                    continue
                out.append((full, f"{prefix}{rel}"))
    return out


# --- Parse worker ---------------------------------------------------------


# The pool worker lives in parse.py so spawned workers import tree-sitter
# only; the name is kept for callers that import it from here.
from docgraph.parse import parse_worker as _parse_worker  # noqa: E402


_SLIM_DROP = ("body", "signature")


def _slim_entity(e: dict) -> dict:
    """Cache form of a parsed entity: everything except the body/signature
    (those are in the DB) and transient extras."""
    out = {k: v for k, v in e.items() if k not in _SLIM_DROP}
    ex = out.get("extra")
    if isinstance(ex, dict) and ("_id" in ex or "llm_doc" in ex):
        out["extra"] = {k: v for k, v in ex.items() if k not in ("_id", "llm_doc")}
    return out


def _terms(*parts: str) -> str:
    """Identifier-split keyword text for the FTS index (`fetchAllRows` ->
    `fetch all rows`); Kuzu's own tokenizer does not split camelCase."""
    from docgraph.bm25 import tokenize
    seen: dict[str, None] = {}
    for p in parts:
        for t in tokenize(p or ""):
            seen.setdefault(t, None)
    return " ".join(seen)


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    try:
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


# --- Cache ----------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a temp file + atomic rename, so a process
    killed mid-write (reaper, crash, power loss) can never leave a truncated
    file. A torn file would read back as garbage and force a full reindex ->
    force a full reindex of everything. os.replace is atomic on Windows
    and POSIX."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --- Indexer --------------------------------------------------------------


class _Stages:
    """Wall-clock per index stage (ms in the result / state.json)."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, dt: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + dt

    def ms(self) -> dict[str, float]:
        return {k: round(v * 1000.0, 1) for k, v in self.t.items()}


class _InlineExecutor:
    """Executor.map look-alike that runs in the calling process."""

    @staticmethod
    def map(fn, items, chunksize: int = 1):
        return map(fn, list(items))


class Indexer:
    INPROC_PARSE_MAX = 8
    KEPT_POOL_MAX = 2000
    KEPT_POOL_WORKERS = 4

    # Incremental SIMILAR_TO for more than this many new entities goes to
    # the host's background maintenance (inline when there is no host).
    SIMILAR_INLINE_MAX = 400

    def _embed_reuse_min(self) -> int:
        """Entities of the changed files above which reading their old
        vectors back (by content hash) beats re-embedding them: ~25 ms warm /
        ~300 ms cold for the harvest vs ~8 ms per entity on CPU, ~1 ms on GPU."""
        dev = str(getattr(self.embedder, "device", "") or "")
        return 64 if dev.startswith("cuda") else 4
    # Watcher event paths replace the directory walk for this long after the
    # last walk; then one walked pass re-validates (missed events, renames).
    WATCH_TRUST_SEC = 600.0

    def __init__(self, cfg: Config, db: GraphDB, embedder: Embedder | None = None,
                 live: LiveIndex | None = None, tile_sink=None, defer_global: bool = False,
                 tile_patcher=None):
        self.cfg = cfg
        self.db = db
        # Incremental state (cache shards, symbol table, import / ref
        # indexes, keyword index, tiles). The host passes the root's
        # long-lived one; the CLI gets a fresh one per run.
        self.live = live if live is not None else LiveIndex(cfg)
        # host hooks: tile_sink(manifest, arrays) publishes a tile generation
        # (served from memory, persisted in the background); defer_global
        # hands big-graph global analytics / bulk SIMILAR_TO to maintenance.
        self.tile_sink = tile_sink
        self.defer_global = bool(defer_global)
        # host hook: tile_patcher(delta, generation) applies an incremental
        # tile patch off the index pass (in order, on the host's persist
        # thread); without it the pass patches inline.
        self.tile_patcher = tile_patcher
        # host hook: entity vectors of incremental passes are returned in
        # `pending_vectors` ({label: [{id, embedding}]}) instead of written
        self.defer_vectors = False
        self.pending_vectors: dict[str, list[dict]] = {}
        self._tiles_full = False
        self._new_test_ids: set[int] = set()
        # vectors of the entities written by this (small, incremental) pass,
        # so SIMILAR_TO does not read them back from the embedding column
        self._fresh_vecs: dict[str, dict[int, object]] | None = None
        self.embedder = embedder or Embedder(
            cfg.embedding_model,
            device=resolve_device(cfg.gpu),
            torch_compile=cfg.embed_torch_compile,
        )
        self._next_id = 1
        self.progress_cb: ProgressCb = None
        self.embed_cache_hits = 0
        self.embed_cache_misses = 0

    # ---- ID allocation ----
    def _seed_ids_from_db(self) -> None:
        """Continue allocating after the max id currently in the DB."""
        max_id = 0
        for label in ("File", "Module", "Class", "Function", "Variable", "Chunk",
                      "Community", "Route", "Tool"):
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

    def _stream_embed_insert(
        self,
        label: str,
        plan: list[tuple[dict, str]],
        prog_label: str,
        insert_batch: int = 5000,
        cancel_token: "CancelToken | None" = None,
        progress_emit: "Callable[[int], None] | None" = None,
        vec_cache: "dict[str, list] | None" = None,
        db_lookup: bool = False,
        prog=None,
        task=None,
    ) -> None:
        """Stream-embed and insert. plan: [(row_dict, embed_text), ...].

        Embedding cache: rows carry `ehash` (model + embed-text hash). A
        row whose hash is in `vec_cache` (vectors harvested from the nodes
        this pass deleted -- i.e. moved / renamed / re-parsed entities) or,
        with `db_lookup`, already stored on another node, reuses that
        vector instead of being re-embedded.

        Memory contract: at any moment we hold the source `plan` plus at most
        two batches of vectors (one being embedded, one in the writer queue).
        Embedding text strings are dropped per-batch as they're consumed.

        A daemon writer thread overlaps Kuzu I/O with the next batch's
        embedding — this is the closest we get to true pipelining without
        rewriting Kuzu's pybind. Bounded queue (maxsize=2) caps in-flight
        batches so a slow writer can't let the embedder run away with RAM.
        """
        if not plan:
            return
        write_q: queue.Queue = queue.Queue(maxsize=2)
        errors: list[BaseException] = []

        def writer() -> None:
            while True:
                item = write_q.get()
                try:
                    if item is None:
                        return
                    self.db.insert_nodes(label, item, batch_size=insert_batch)
                except BaseException as exc:  # noqa: BLE001 - propagate to main thread
                    errors.append(exc)
                    return
                finally:
                    write_q.task_done()

        t = threading.Thread(target=writer, daemon=True, name=f"writer-{label}")
        t.start()
        try:
            total = len(plan)
            own = prog is None
            with (_bar() if own else contextlib.nullcontext(prog)) as prog:
                if own:
                    task = prog.add_task(prog_label, total=total)
                # Pop slabs off the front so already-consumed rows + body
                # strings can be GC'd while later batches are still embedding.
                # Caller's list is mutated; callers .clear() it anyway.
                while plan:
                    if errors:
                        break
                    # Per-batch cancel checkpoint: between batches is
                    # safe (last batch's writes already committed via
                    # the writer thread's queue handoff).
                    if cancel_token is not None:
                        cancel_token.raise_if_set()
                    batch = plan[:insert_batch]
                    del plan[:insert_batch]
                    texts = [t for _, t in batch]
                    rows = [r for r, _ in batch]
                    del batch
                    def _on_emb(n: int) -> None:
                        if cancel_token is not None:
                            cancel_token.raise_if_set()
                        prog.advance(task, n)
                        if progress_emit is not None:
                            try:
                                progress_emit(n)
                            except Exception:
                                pass
                    todo = list(range(len(rows)))
                    if vec_cache is not None:
                        if db_lookup:
                            missing = [rows[i].get("ehash") for i in todo
                                       if rows[i].get("ehash") and rows[i]["ehash"] not in vec_cache]
                            if missing:
                                try:
                                    vec_cache.update(self.db.embeddings_by_ehash(label, missing))
                                except Exception:
                                    log.debug("embedding cache lookup failed", exc_info=True)
                        hit = [i for i in todo if rows[i].get("ehash") in vec_cache]
                        for i in hit:
                            rows[i]["embedding"] = vec_cache[rows[i]["ehash"]]
                        if hit:
                            self.embed_cache_hits += len(hit)
                            _on_emb(len(hit))
                        hit_set = set(hit)
                        todo = [i for i in todo if i not in hit_set]
                    self.embed_cache_misses += len(todo)
                    if todo:
                        vecs = self.embedder.embed(
                            [texts[i] for i in todo],
                            batch_size=self.cfg.embed_batch_size,
                            on_progress=_on_emb,
                        )
                        # Attach numpy slices (1.5 KB each) instead of list[float]
                        # (12 KB each) — db.insert_nodes converts to list per
                        # write batch just before the UNWIND call.
                        for i, v in zip(todo, vecs):
                            rows[i]["embedding"] = v
                        del vecs
                    if self._fresh_vecs is not None:
                        fv = self._fresh_vecs.setdefault(label, {})
                        for r in rows:
                            v = r.get("embedding")
                            if v is not None:
                                fv[int(r["id"])] = v
                    write_q.put(rows)
                    # Free the embedding text strings for this slab.
                    del texts
        finally:
            write_q.put(None)
            t.join()
        if errors:
            raise errors[0]

    def _write_parsed_batch(self, parsed: dict[str, FileParse], new_hashes: dict[str, str],
                            blame_map: dict, vec_cache: dict, use_cache: bool,
                            cancel_token, prog, etask, embed_state: dict,
                            embed_emit: Callable[[int], None], counts: dict) -> dict:
        """Rows for one parsed batch -> File/Variable inserts, embedded
        Class/Function/Chunk inserts and CONTAINS_CHUNK edges. Returns
        {"files": [(rel, file_id, [(label, id, name, qname)])],
         "kw": {label: [(id, term Counter)]}} for the live symbol table and
        the keyword index."""
        from docgraph.kwindex import doc_terms
        file_rows: list[dict] = []
        variable_rows: list[dict] = []
        class_plan: list[tuple[dict, str]] = []
        function_plan: list[tuple[dict, str]] = []
        chunk_plan: list[tuple[dict, str]] = []
        cc_func: list[dict] = []
        cc_class: list[dict] = []
        cc_file: list[dict] = []
        model = self.embedder.model_name
        for rel, fp in parsed.items():
            fid = self._new_id()
            file_rows.append({
                "id": fid, "path": rel, "language": fp.language, "lines": fp.lines,
                "hash": new_hashes[rel], "pagerank": 0.0,
            })
            for ent in fp.entities:
                eid = self._new_id()
                ent.extra["_id"] = eid
                llm_doc = ent.extra.get("llm_doc") if isinstance(ent.extra, dict) else None
                if ent.kind in ("class", "interface", "function", "method"):
                    text = build_embedding_text(ent.name, ent.qname, ent.signature, ent.body,
                                                fp.language, ent.kind, llm_doc=llm_doc)
                    row = {
                        "id": eid, "name": ent.name, "qname": ent.qname, "file": rel,
                        "line_start": ent.line_start, "line_end": ent.line_end,
                        "body": ent.body, "llm_doc": llm_doc, "pagerank": 0.0,
                        "terms": _terms(ent.name, ent.qname.replace("::", " ")),
                    }
                    if ent.kind in ("class", "interface"):
                        row["kind"] = ent.kind
                    else:
                        row["signature"] = ent.signature or ent.body.split("\n")[0][:200]
                        row["is_method"] = ent.kind == "method"
                        row["is_test"] = (
                            ent.name.startswith("test_") or
                            (ent.name.startswith("test") and len(ent.name) > 4 and ent.name[4:5].isupper()) or
                            "/test" in rel or "/tests/" in rel or "_test." in rel
                        )
                    row["ehash"] = _ehash(model, text)
                    row.update(self._history_cols(blame_map.get(rel), ent.line_start, ent.line_end))
                    (class_plan if ent.kind in ("class", "interface") else function_plan).append((row, text))
                else:
                    variable_rows.append({
                        "id": eid, "name": ent.name, "qname": ent.qname, "file": rel,
                        "line": ent.line_start,
                        "scope": ent.extra.get("scope", "module") if isinstance(ent.extra, dict) else "module",
                    })
            # Sub-entity chunks (long bodies) ...
            for ent in fp.entities:
                eid = ent.extra.get("_id") if isinstance(ent.extra, dict) else None
                if eid is None or ent.kind not in ("function", "method", "class", "interface"):
                    continue
                body = ent.body or ""
                pieces = chunk_body(body, language=fp.language)
                if not pieces:
                    continue
                parent_label = "Class" if ent.kind in ("class", "interface") else "Function"
                pos = 0
                for idx, piece in enumerate(pieces):
                    cid = self._new_id()
                    at = body.find(piece[:120], pos)
                    if at < 0:
                        at = pos
                    pos = at + 1
                    ls = ent.line_start + body.count("\n", 0, at)
                    le = ls + piece.count("\n")
                    body_truncated = piece[:6000]
                    row = {
                        "id": cid, "parent_qname": ent.qname, "parent_label": parent_label,
                        "file": rel, "idx": idx, "body": body_truncated,
                        "ehash": _ehash(model, body_truncated),
                        "line_start": ls, "line_end": min(le, ent.line_end),
                        "terms": _terms(ent.name, rel),
                    }
                    chunk_plan.append((row, body_truncated))
                    (cc_func if parent_label == "Function" else cc_class).append({"from_id": eid, "to_id": cid})
            # ... and file-level chunks (plain text / symbol-less / notebook markdown)
            for ch in fp.chunks or []:
                cid = self._new_id()
                body = str(ch.get("body") or "")[:6000]
                text = f"{rel}\n{body}"
                chunk_plan.append(({
                    "id": cid, "parent_qname": rel, "parent_label": "File", "file": rel,
                    "idx": int(ch.get("idx", 0)), "body": body, "ehash": _ehash(model, text),
                    "line_start": int(ch.get("line_start") or 1),
                    "line_end": int(ch.get("line_end") or ch.get("line_start") or 1),
                    "terms": _terms(rel),
                }, text))
                cc_file.append({"from_id": fid, "to_id": cid})

        counts["files"] += len(file_rows)
        counts["variables"] += len(variable_rows)
        counts["classes"] += len(class_plan)
        counts["functions"] += len(function_plan)
        counts["chunks"] += len(chunk_plan)
        if file_rows:
            self.db.insert_nodes("File", file_rows)
        if variable_rows:
            self.db.insert_nodes("Variable", variable_rows)
        # live symbol table + keyword docs, from the rows about to be written
        ents_by_file: dict[str, list] = defaultdict(list)
        for r in variable_rows:
            ents_by_file[r["file"]].append(("Variable", r["id"], r["name"], r["qname"]))
        kw: dict[str, list] = {"Function": [], "Class": [], "Chunk": []}
        for label, plan in (("Class", class_plan), ("Function", function_plan)):
            for r, _t in plan:
                ents_by_file[r["file"]].append((label, r["id"], r["name"], r["qname"]))
                kw[label].append((r["id"], doc_terms(r["name"], r.get("terms") or "", r.get("body") or "")))
        for r, _t in chunk_plan:
            kw["Chunk"].append((r["id"], doc_terms(r.get("terms") or "", r.get("body") or "")))
        out = {"files": [], "kw": kw}
        fid_of = {r["path"]: r["id"] for r in file_rows}
        for rel in parsed:
            ents = sorted(ents_by_file.get(rel, ()), key=lambda e: e[1])
            out["files"].append((rel, fid_of.get(rel), ents))
        n_embed = len(class_plan) + len(function_plan) + len(chunk_plan)
        if n_embed:
            embed_state["total"] += n_embed
            prog.update(etask, total=embed_state["total"])
            self.embedder._ensure()
        for label, plan, lookup in (("Class", class_plan, use_cache),
                                    ("Function", function_plan, use_cache),
                                    ("Chunk", chunk_plan, False)):
            if plan:
                self._stream_embed_insert(
                    label, plan, f"Embedding ({label})", cancel_token=cancel_token,
                    progress_emit=embed_emit,
                    vec_cache=vec_cache.get(label, {}) if use_cache else None,
                    db_lookup=lookup, prog=prog, task=etask,
                )
                plan.clear()
        self.db.insert_edges("CONTAINS_CHUNK", "Function", "Chunk", cc_func, validate=False)
        self.db.insert_edges("CONTAINS_CHUNK", "Class", "Chunk", cc_class, validate=False)
        self.db.insert_edges("CONTAINS_CHUNK", "File", "Chunk", cc_file, validate=False)
        return out

    # ---- DB delete ----
    def _augment_llm_docstrings(self, parsed: dict,
                                 cancel_token: "CancelToken | None" = None) -> None:
        """For entities lacking a native docstring, ask the local LLM to
        write a one-sentence summary. Cached by body hash in
        `.docgraph/llm_docstrings.json` so incrementals don't re-call.
        Skipped silently if the LLM endpoint is unreachable."""
        import hashlib
        from concurrent.futures import ThreadPoolExecutor

        from docgraph.llm import LLMClient, LLMConfig
        from docgraph.summary import extract_docstring

        cache_path = self.cfg.data_dir / "llm_docstrings.json"
        cache: dict[str, str] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}

        targets: list[tuple[object, str, object]] = []  # (entity, body_hash, fileparse)
        for fp in parsed.values():
            for ent in fp.entities:
                if ent.kind not in ("function", "method", "class", "interface"):
                    continue
                if not ent.body:
                    continue
                if extract_docstring(ent.body, fp.language).strip():
                    continue  # already has a native docstring
                h = hashlib.sha256(ent.body.encode("utf-8", errors="replace")).hexdigest()
                if h in cache:
                    if isinstance(ent.extra, dict):
                        ent.extra["llm_doc"] = cache[h]
                    continue
                targets.append((ent, h, fp))

        if not targets:
            log.info("LLM docstrings: no new entities to augment in this pass")
            return

        log.info("LLM docstrings: augmenting %d entities missing native docstrings...", len(targets))
        client = LLMClient(LLMConfig(
            host=self.cfg.llm_host,
            port=self.cfg.llm_port,
            model=self.cfg.llm_model,
            format=self.cfg.llm_format,
            max_tokens=self.cfg.llm_max_tokens,
            api_key=getattr(self.cfg, "llm_api_key", "") or None,
            timeout=int(getattr(self.cfg, "llm_timeout", 1800) or 1800),
        ))

        def _task(item):
            ent, h, fp = item
            text = client.summarize(ent.kind, ent.name, ent.body, fp.language)
            return ent, h, text

        n_workers = min(8, max(2, self.cfg.workers))
        total = len(targets)
        done = 0

        # We use a context manager for the Rich bar if we're in a CLI (no cb)
        # but if we have a cb, we just use it directly.
        with _bar() if not self.progress_cb else contextlib.nullcontext() as prog:
            ptask = prog.add_task("LLM docstrings", total=total) if prog else None
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for ent, h, text in ex.map(_task, targets):
                    if cancel_token is not None:
                        cancel_token.raise_if_set()
                    if text:
                        if isinstance(ent.extra, dict):
                            ent.extra["llm_doc"] = text
                        cache[h] = text
                    
                    done += 1
                    if prog and ptask is not None:
                        prog.advance(ptask)
                    if self.progress_cb:
                        self.progress_cb("llm_augment", done, total)

        log.info("LLM docstrings: augmentation complete (%d processed)", total)
        try:
            _atomic_write_text(cache_path, json.dumps(cache))
        except Exception:
            pass

    def index_all(self, incremental: bool = True, progress_cb: ProgressCb = None,
                  cancel_token: "CancelToken | None" = None,
                  fetch_links: bool = True, force_fetch: bool = False,
                  changed_paths: "list[str | Path] | None" = None) -> dict:
        """Index the root. Incremental passes are proportional to the change:
        the scan reuses cached decisions (or takes `changed_paths` from the
        watcher and skips the walk), the cache is sharded (only dirty shards
        are written), the symbol table / import records / reverse refs live
        in `self.live` and are patched per file, and only the edges that can
        reach a changed file are resolved (see _resolve_scope)."""
        self.progress_cb = progress_cb
        T = _Stages()
        # Cooperative cancel: poll `cancel_token.raise_if_set()` at major
        # phase boundaries. Mid-phase cancellation is unsafe (inside Kuzu
        # COPY or a torch forward pass would corrupt state); between phases
        # is fine because each phase commits before the next starts.
        def _ck():
            if cancel_token is not None:
                cancel_token.raise_if_set()

        # Progress emit — fires the user-supplied callback at each phase
        # boundary so external supervisors (telecode SSE) can mirror the
        # same status the CLI prints. Callback signature: (phase, current,
        # total). For phases without a count both are 0. Wraps in try so a
        # broken callback can't kill the index pass.
        def _emit(phase: str, current: int = 0, total: int = 0) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(phase, current, total)
            except Exception:
                log.debug("progress_cb raised; ignoring", exc_info=True)

        # Throttled per-item emitter. Long phases (parse 30k files, embed
        # 170k entities) would otherwise dump thousands of SSE events; cap
        # at one update per second + always fire on completion.
        def _throttled(phase: str, total: int):
            state = {"done": 0, "last": 0.0}
            def push(n: int = 1) -> None:
                state["done"] += n
                now = time.perf_counter()
                if now - state["last"] >= 1.0 or state["done"] >= total:
                    state["last"] = now
                    _emit(phase, state["done"], total)
            return push

        with T("links"):
            _wire_extra_paths(self.cfg)
            if fetch_links:
                _emit("fetch_links")

                def _fetch_progress(depth: int, done: int, total: int) -> None:
                    _emit(f"fetch:{depth}", done, total)

                _maybe_fetch_links(self.cfg, force=force_fetch,
                                   cancel_check=cancel_token.raise_if_set if cancel_token is not None else None,
                                   progress_cb=_fetch_progress)

        _ck()
        _emit("start")
        t0 = time.perf_counter()
        live = self.live
        state0 = self._load_state()
        with T("cache_load"):
            if incremental:
                live.ensure_cache()
        cache = live.cache
        if incremental and state0.get("schema_version") != SCHEMA_VERSION:
            # The DB schema or the cache entry shape changed since this root
            # was indexed. An incremental pass would mix both shapes, so
            # rebuild once.
            if cache or state0:
                _console.print(
                    f"[yellow]Index format v{state0.get('schema_version', 1)} -> "
                    f"v{SCHEMA_VERSION}: running a full reindex[/]"
                )
                log.warning("schema v%s -> v%s: forcing full reindex of %s",
                            state0.get("schema_version", 1), SCHEMA_VERSION, self.cfg.repo_root)
            incremental = False
        if incremental and not cache:
            incremental = False          # no per-file state: a delta cannot be computed
        cache_was_present = bool(cache) and incremental
        if not incremental:
            live.reset()
            cache = live.cache

        # ---- scan: which files changed? ----
        with T("scan"):
            fast = (incremental and changed_paths is not None and live.scan_ready
                    and time.time() - live.last_walk < self.WATCH_TRUST_SEC)
            candidates: dict[str, tuple[Path, int, int]]
            if fast:
                present, gone = classify_paths(self.cfg, live.scan_state, changed_paths, MAX_FILE_BYTES)
                deleted_rels = sorted({g for g in gone if g in cache} |
                                      {r for g in gone if g not in cache for r in cache
                                       if r.startswith(g.rstrip("/") + "/")})
                candidates = present
                n_files = len(cache) - len(deleted_rels) + sum(1 for r in present if r not in cache)
            else:
                candidates = live_scan(self.cfg, live.scan_state, MAX_FILE_BYTES)
                deleted_rels = sorted(r for r in cache if r not in candidates)
                n_files = len(candidates)
                live.scan_ready = True
                live.last_walk = time.time()
            changed: list[tuple[Path, str]] = []
            new_hashes: dict[str, str] = {}
            new_stat: dict[str, tuple[int, int]] = {}
            hashed = reused = 0
            for rel, (path, size, mtime) in candidates.items():
                cm = cache.meta(rel) if incremental else None
                if cm is not None and cm[0] == size and cm[1] == mtime:
                    reused += 1
                    continue
                h = _file_hash(path)
                hashed += 1
                new_hashes[rel] = h
                new_stat[rel] = (size, mtime)
                if cm is not None and cm[2] == h:
                    cache.set_stat(rel, size, mtime)   # stat refresh only (marks the shard dirty)
                    continue
                changed.append((path, rel))
        _console.print(
            f"[cyan]Scanning[/]: {n_files} files — "
            f"[green]{len(changed)}[/] changed/added, "
            f"[red]{len(deleted_rels)}[/] deleted "
            f"[dim](hashed {hashed}, stat-reused {reused}{', watcher paths' if fast else ''})[/]"
        )

        # No changes: bail
        if incremental and not changed and not deleted_rels:
            with T("persist"):
                cache.save()
            hist_updated = self._refresh_pending_history(state0, set())
            if hist_updated:
                self._save_state(state0)
            return {
                "files": n_files, "changed": 0, "deleted": 0,
                "entities": state0.get("entities", 0),
                "elapsed": time.perf_counter() - t0, "errors": 0,
                "hashed": hashed, "hash_reused": reused, "timings": T.ms(),
            }

        # Full reindex path
        if not incremental:
            # Release the Windows file lock before rmtree — Kuzu's connection
            # holds the dir open and shutil.rmtree silently leaves a partial
            # state, which then crashes the next Database() constructor with
            # `invalid unordered_map<K, T> key`.
            with T("wipe"):
                self.db.close()
                self.db.wipe(self.cfg.db_path)
                self.db = GraphDB(self.cfg.db_path, self.embedder.dim)
                self.db.init_schema()
                self._next_id = 1
                cache.wipe()
                for legacy in (self.cfg.cache_path, self.cfg.data_dir / "merkle.json"):
                    try:
                        legacy.unlink()
                    except OSError:
                        pass
            deleted_rels = []
            changed = [(p, rel) for rel, (p, _s, _m) in candidates.items()]

        symtab = live.symtab
        with T("symtab"):
            if incremental:
                symtab = live.ensure_symtab(self.db)
        affected = [rel for _path, rel in changed]
        gone_all = affected + deleted_rels
        prev_files = set(symtab.file_index)

        # ---- harvest + delete the old nodes of changed / deleted files ----
        _ck()
        _emit("delete", 0, len(gone_all))
        vec_cache: dict[str, dict[str, list]] = {}
        old_entities: dict[str, list[tuple[str, str, str]]] = {}
        harvest: dict = {}
        old_ids: dict[str, list[int]] = {}
        chunk_ids: list[int] = []
        before_cands: dict[str, frozenset] = {}
        incoming: dict | None = None
        reuse_vectors = False
        with T("harvest"):
            if incremental and gone_all:
                old_ids = symtab.ids_of_files(gone_all)
                chunk_ids = self.db.ids_in_files("Chunk", gone_all)
                # Vectors of the deleted rows are reused (by content hash) only
                # when there are enough of them to pay for reading the
                # embedding column: re-embedding a handful of entities is
                # cheaper than the harvest query.
                n_old = len(old_ids.get("Function", [])) + len(old_ids.get("Class", [])) + len(chunk_ids)
                reuse_vectors = bool(getattr(self.cfg, "embed_cache", True)) and n_old >= self._embed_reuse_min()
                vec_ids, vec_chunks = old_ids, chunk_ids
                if reuse_vectors:
                    for label in ("Function", "Class"):
                        try:
                            vec_cache[label] = self.db.embeddings_by_ids(label, vec_ids.get(label, []))
                        except Exception:
                            vec_cache[label] = {}
                    try:
                        vec_cache["Chunk"] = self.db.embeddings_by_ids("Chunk", vec_chunks)
                    except Exception:
                        vec_cache["Chunk"] = {}
                harvest = self._harvest_layout_ids(old_ids, gone_all)
                # edges from unchanged code into the nodes about to be
                # recreated: re-pointed at the new ids (no re-resolution)
                try:
                    incoming = {"edges": self.db.incoming_edges(old_ids),
                                "keys": {i: (name, qn) for rel in gone_all
                                         for _lab, i, name, qn in symtab.by_file.get(rel, ())}}
                    for rel in gone_all:
                        fid = symtab.file_index.get(rel)
                        if fid is not None:
                            incoming["keys"][fid] = (rel, rel)
                except Exception:
                    log.debug("incoming edge harvest failed", exc_info=True)
                    incoming = None
                # candidate sets of every name the change touches, before it
                q = symtab.id_qname
                for rel in gone_all:
                    for _lab, _i, name, _qn in symtab.by_file.get(rel, ()):
                        if name not in before_cands:
                            before_cands[name] = frozenset(
                                (c[2], q.get(c[1], ""), c[0]) for c in symtab.name_index.get(name, ()))
            for rel in gone_all:
                ents = (cache.get(rel) or {}).get("entities") or []
                if ents:
                    old_entities[rel] = [(e.get("qname", ""), e.get("name", ""), e.get("kind", ""))
                                         for e in ents]
        # every graph write from here on is journalled for the tile patch
        self.db.journal = EdgeJournal() if incremental else None
        self._tiles_full = False
        # host: the vectors of the entities this pass writes go to background
        # maintenance (the HNSW insertion is the costliest write of a pass)
        self.pending_vectors = {}
        if incremental and self.defer_vectors:
            def _vsink(label, rows, _pv=self.pending_vectors):
                _pv.setdefault(label, []).extend(rows)
            self.db.vector_sink = _vsink
        with T("delete"):
            if incremental and gone_all:
                live.remove_tests(old_ids.get("Function", []))
                kw = live.ensure_kw()
                for label in ("Function", "Class"):
                    kw.remove(label, old_ids.get(label, []))
                kw.remove("Chunk", chunk_ids)
                for label in ("Function", "Class", "Variable"):
                    self.db.delete_by_ids(label, old_ids.get(label, []))
                for label in ("Route", "Tool"):
                    if self.db.has_table(label):
                        self.db.execute(f"MATCH (n:{label}) WHERE n.file IN $files DETACH DELETE n",
                                        {"files": gone_all})
                self.db.delete_by_ids("Chunk", chunk_ids)
                self.db.delete_by_ids("File", old_ids.get("File", []))
                symtab.remove_files(gone_all)
                if live.resolve_ready:
                    for rel in gone_all:
                        live.imports.remove(rel)
                        live.refs.remove(rel)
            for rel in deleted_rels:
                cache.pop(rel, None)

        # ---- Steps 2-5: parse -> rows -> embed -> insert, streamed in file batches ----
        # Memory stays flat on big repos: each batch of `index_batch_files`
        # files is parsed, blamed, (optionally) LLM-augmented, turned into
        # node rows, embedded and written before the next batch is parsed, so
        # at most one batch of bodies / vectors is alive at a time.
        _ck()
        _emit("seed_ids")
        if incremental and live.next_id:
            self._next_id = int(live.next_id)
        elif incremental:
            self._seed_ids_from_db()
        errors: list[str] = []
        changed_set: set[str] = set()
        n_parsed = 0
        use_cache = incremental and reuse_vectors
        text_fallback = bool(getattr(self.cfg, "text_fallback", True))
        batch_files = max(50, int(getattr(self.cfg, "index_batch_files", 2000) or 2000))
        blame_budget = int(getattr(self.cfg, "history_max_files", 5000) or 0)
        counts = {"files": 0, "classes": 0, "functions": 0, "variables": 0, "chunks": 0}
        new_pending: set[str] = set()
        kw_builder = None if incremental else KeywordBuilder()
        self._fresh_vecs = {} if incremental and len(changed) <= 2000 else None
        kw_live = live.ensure_kw() if incremental else None
        # A large incremental (a branch switch, a big merge) reloads like a
        # full pass: drop the vector indexes so inserts skip HNSW upkeep, and
        # let ensure_search_indexes() rebuild them afterwards.
        if incremental and len(changed) > max(2000, 0.1 * max(1, n_files)):
            self.db.drop_search_indexes()
            live.vector_ok = False
        t_parse = time.perf_counter()
        if changed:
            _emit("parse", 0, len(changed))
            parse_emit = _throttled("parse", len(changed))
            roots = self.cfg.roots_with_prefix()
            args_all: list[tuple[str, str, str, bool]] = []
            for path, logical_rel in changed:
                owner = self.cfg.repo_root
                for root, prefix in roots:
                    if prefix == "" or logical_rel.startswith(prefix):
                        owner = root
                        break
                args_all.append((str(path), str(owner), logical_rel, text_fallback))
            embed_state = {"done": 0, "total": 0, "last": 0.0}

            def _embed_emit(n: int) -> None:
                embed_state["done"] += n
                now = time.perf_counter()
                if now - embed_state["last"] >= 1.0:
                    embed_state["last"] = now
                    _emit("embed_entities", embed_state["done"], embed_state["total"])

            # A handful of files (the watcher's usual case) parse in-process:
            # spawning the pool costs more than the parse. The host keeps one
            # pool warm between passes (`LiveIndex.parse_pool`) for changes
            # up to KEPT_POOL_MAX files; bigger runs get their own full pool.
            n_workers = max(1, min(int(self.cfg.workers or 1), len(args_all)))
            if len(args_all) <= self.INPROC_PARSE_MAX:
                pool_cm = contextlib.nullcontext(_InlineExecutor())
            elif getattr(self.live, "keep_pool", False) and len(args_all) <= self.KEPT_POOL_MAX:
                pool_cm = contextlib.nullcontext(
                    self.live.parse_pool(max(1, min(int(self.cfg.workers or 1), self.KEPT_POOL_WORKERS))))
            else:
                pool_cm = ProcessPoolExecutor(max_workers=n_workers)
            with _bar() as prog, pool_cm as ex:
                ptask = prog.add_task("Parsing files", total=len(changed))
                etask = prog.add_task("Embedding", total=0)
                # Executor.map submits eagerly: queueing batch N+1 before
                # batch N is embedded keeps the pool parsing while the GPU
                # embeds (at most two batches of parse results in flight).
                def _submit(start: int):
                    return ex.map(_parse_worker, args_all[start:start + batch_files], chunksize=8)

                pending = _submit(0)
                for b0 in range(0, len(args_all), batch_files):
                    _ck()
                    current = pending
                    nxt = b0 + batch_files
                    pending = _submit(nxt) if nxt < len(args_all) else None
                    parsed: dict[str, FileParse] = {}
                    for result in current:
                        prog.advance(ptask)
                        parse_emit(1)
                        # Per-file checkpoint: a cancel lands within a few
                        # hundred ms instead of after the whole pool drains.
                        _ck()
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
                            chunks=result.get("chunks") or [],
                            extra=result.get("extra") or {},
                        )
                        parsed[fp.file] = fp
                        size, mtime = new_stat.get(fp.file, (0, 0))
                        cache[fp.file] = {
                            "hash": new_hashes.get(fp.file, ""),
                            "size": size, "mtime": mtime,
                            "language": fp.language,
                            "lines": fp.lines,
                            # Bodies / signatures live in the DB; the cache
                            # only needs what resolution and the delta read.
                            "entities": [_slim_entity(e) for e in result["entities"]],
                            "edges": result["edges"],
                        }
                    if not parsed:
                        continue
                    # 2b: symbol history (one `git blame` per parsed file)
                    blame_map: dict[str, list[tuple[str, int]]] = {}
                    if getattr(self.cfg, "history", True) and blame_budget > 0:
                        _ck()
                        _emit("history", 0, len(parsed))
                        try:
                            jobs = self._blame_jobs(list(parsed.keys()))
                            if blame_budget > 0:
                                jobs = jobs[:blame_budget]
                                blame_budget -= len(jobs)
                            blame_map = _history.blame_many(
                                jobs, workers=min(8, max(1, self.cfg.workers)),
                                max_files=len(jobs) or 1,
                            )
                        except Exception:
                            log.debug("history blame failed", exc_info=True)
                            blame_map = {}
                    new_pending.update(rel for rel, b in blame_map.items()
                                       if any(sha == _history.UNCOMMITTED for sha, _ in b))
                    # 3a: optional LLM docstring augmentation
                    _ck()
                    if self.cfg.llm_docstrings:
                        _emit("llm_augment", 0, len(parsed))
                        self._augment_llm_docstrings(parsed, cancel_token=cancel_token)
                    # 4-5: rows, embeddings, inserts for this batch
                    out = self._write_parsed_batch(
                        parsed, new_hashes, blame_map, vec_cache, use_cache,
                        cancel_token, prog, etask, embed_state, _embed_emit, counts,
                    )
                    for rel, fid, ents in out["files"]:
                        symtab.add_file(rel, fid, ents, sort=incremental)
                    for label, docs in out["kw"].items():
                        if kw_builder is not None:
                            kw_builder.add(label, docs)
                        else:
                            kw_live.add(label, docs)
                    changed_set.update(parsed.keys())
                    n_parsed += len(parsed)
                    parsed.clear()
            _emit("embed_entities", embed_state["done"], embed_state["total"])
        T.add("parse_embed_insert", time.perf_counter() - t_parse)
        if not incremental:
            symtab.finish_bulk()
            symtab.loaded = True
            with T("keyword_index"):
                kw_builder.finish(live.ensure_kw(load=False))
        _console.print(
            f"[cyan]Indexed[/] {counts['files']} files, {counts['classes']} classes, "
            f"{counts['functions']} functions, {counts['variables']} variables, "
            f"{counts['chunks']} chunks"
        )
        del vec_cache

        # Search indexes (HNSW vectors). A fresh DB gets them once, after the
        # bulk load; an existing DB keeps them current by itself.
        _ck()
        _emit("search_index")
        with T("search_index"):
            if incremental and live.vector_ok:
                search_status = live.vector_status
            else:
                search_status = self.db.ensure_search_indexes(
                    on_progress=lambda what: _console.print(f"[dim]Building {what}[/]"))
                live.vector_status = search_status
                live.vector_ok = all(v == "ok" for v in search_status.get("vector", {}).values())

        # ---- Step 7-8: scoped resolution + edge writes ----
        _ck()
        _emit("symbol_table")
        _emit("edges")
        with T("resolve"):
            res = self._resolve_scope(live, cache, changed_set, deleted_rels, incremental,
                                      prev_files, before_cands, incoming)
        with T("edge_write"):
            self._write_resolved(res)
        resolution_stats = res["stats"]
        scope_info = res["scope"]
        live.next_id = self._next_id

        # ---- Step 8a: optional precise references (SCIP) ----
        scip_status: dict = {}
        with T("scip"):
            try:
                from docgraph import scip as _scip
                _ck()
                scip_status = _scip.maybe_ingest(self.cfg, self.db, full=not cache_was_present,
                                                 changed=bool(changed_set) or bool(deleted_rels),
                                                 console=_console)
            except Exception as exc:  # noqa: BLE001 - never fail the index over SCIP
                log.warning("SCIP ingest failed: %s", exc)
                scip_status = {"status": "error", "detail": str(exc)}
        if scip_status.get("status") == "ingested" or scip_status.get("edges"):
            self._tiles_full = True       # SCIP rewrites REFERENCES_ wholesale

        # ---- Step 8c: history of unchanged files blamed while uncommitted ----
        state = state0
        pending_before = set(state.get("history_pending") or [])
        removed = list(state.get("removed_symbols") or [])
        now_ts = time.time()
        head_now = self._primary_head() if old_entities else None
        for rel, olds in old_entities.items():
            new_q = set()
            if rel in cache:
                new_q = {e.get("qname") for e in cache[rel].get("entities", [])}
            for qn, nm, kd in olds:
                if qn and qn not in new_q and kd in ("function", "method", "class", "interface"):
                    removed.append({"qname": qn, "name": nm, "kind": kd, "file": rel,
                                    "removed_at": now_ts, "head": head_now or ""})
        removed = removed[-MAX_REMOVED_SYMBOLS:]

        # ---- Step 8b: LINKS_TO edges from BFS web crawl ----
        # page_links.json is written by fetch_all whenever pages are crawled.
        # External files have path "external/<filename>" in the file index
        # because external_dir (name="external") is appended to
        # cfg.extra_roots before the scan, giving it the prefix "external/".
        file_index = symtab.file_index
        _page_links_file = self.cfg.external_dir / "page_links.json"
        _ext_prefix = self.cfg.external_dir.name + "/"
        links_dirty = (not cache_was_present) or any(
            r.startswith(_ext_prefix) for r in list(changed_set) + list(deleted_rels))
        if links_dirty and _page_links_file.exists() and file_index:
            with T("links_to"):
                try:
                    _link_data = json.loads(_page_links_file.read_text(encoding="utf-8"))
                    links_to_rows: list[dict] = []
                    for _e in _link_data:
                        _fid = file_index.get(_ext_prefix + _e.get("from", ""))
                        _tid = file_index.get(_ext_prefix + _e.get("to", ""))
                        if _fid is not None and _tid is not None and _fid != _tid:
                            links_to_rows.append({"from_id": _fid, "to_id": _tid})
                    if links_to_rows:
                        self.db.delete_all_edges("LINKS_TO")
                        self.db.insert_edges("LINKS_TO", "File", "File", links_to_rows, validate=False)
                        log.info("LINKS_TO: inserted %d hyperlink edges", len(links_to_rows))
                except Exception as _exc:
                    log.warning("LINKS_TO: failed to load page_links.json: %s", _exc)

        _ck()
        _emit("tier4_pagerank")
        # ---- Step 9: Tier 4 + analytics (incremental-aware) ----
        graph_dirty = bool(changed_set) or bool(deleted_rels)
        full_recompute = not cache_was_present
        n_communities = None
        if not graph_dirty and not full_recompute:
            _console.print("[dim]Tier 4 + PageRank: no changes — skipped[/]")
        else:
            with T("tier4"):
                self._recompute_tier4(
                    changed_files=changed_set,
                    deleted_files=set(deleted_rels),
                    full=full_recompute,
                    state=state,
                )
            _ck()
            _emit("analytics")
            with T("analytics"):
                try:
                    info = self._graph_analytics(
                        full=full_recompute, changed=changed_set, deleted=set(deleted_rels),
                        n_files=n_files, state=state, harvest=harvest or {},
                        emit=lambda ph: _emit(ph), removed_ids=old_ids, chunk_ids=chunk_ids)
                    n_communities = info.get("communities")
                    state.pop("analytics_error", None)
                except Exception as exc:  # noqa: BLE001 - never fail the index over analytics
                    log.warning("graph analytics failed: %s", exc, exc_info=True)
                    state["analytics_error"] = str(exc)
        self.db.journal = None
        self.db.vector_sink = None

        # History: files blamed while they had uncommitted lines get
        # re-blamed once HEAD has moved (a commit does not change their hash).
        state["history_pending"] = sorted(pending_before | new_pending)
        self._refresh_pending_history(state, changed_set)
        state["removed_symbols"] = removed

        # Persist state (last-known git HEAD, last_indexed_at, etc.)
        state["last_indexed_at"] = time.time()
        state["schema_version"] = SCHEMA_VERSION
        state["embedding_model"] = self.embedder.model_name
        state["resolution"] = resolution_stats
        state["embed_cache"] = {"hits": self.embed_cache_hits, "misses": self.embed_cache_misses}
        state["search_index"] = search_status
        state["resolution_scope"] = scope_info
        state["scan"] = {"hashed": hashed, "reused": reused, "watcher_paths": bool(fast)}
        state["entities"] = len(symtab.id_qname)
        state["files"] = n_files
        if n_communities is not None:
            state["communities"] = n_communities
        if scip_status:
            state["scip"] = scip_status
        state["timings_ms"] = T.ms()

        # ---- Step 10: persist (dirty cache shards, keyword index, state) ----
        with T("persist"):
            n_shards = cache.save()
            live.ensure_kw(load=False).save()
            state["timings_ms"] = T.ms()
            self._save_state(state)
        live.generation += 1
        live.next_id = self._next_id

        elapsed = time.perf_counter() - t0
        total_entities = len(symtab.id_qname)
        _console.print(
            f"[green]Done[/] in {elapsed:.2f}s — "
            f"{n_files} files, {total_entities} entities, "
            f"{n_parsed} reparsed, {len(deleted_rels)} deleted, {len(errors)} errors "
            f"[dim]({n_shards} cache shards written)[/]"
        )
        _emit("done", n_parsed, n_files)
        return {
            "files": n_files,
            "changed": n_parsed,
            "deleted": len(deleted_rels),
            "entities": total_entities,
            "elapsed": elapsed,
            "errors": len(errors),
            "hashed": hashed,
            "hash_reused": reused,
            "embed_cache_hits": self.embed_cache_hits,
            "embedded": self.embed_cache_misses,
            "communities": n_communities if n_communities is not None else -1,
            "timings": T.ms(),
            "scope": scope_info,
        }

    # ---- scoped resolution ------------------------------------------------
    #
    # A full pass resolves every raw edge. An incremental pass resolves only
    # what the change can affect, in four modes per source file:
    #
    #   normal   files that changed: every edge; an edge is written when
    #            either endpoint was just (re)created (`needs_insert`).
    #   force    unchanged files whose *imports* now resolve differently (a
    #            file they import by name was added / removed): all their
    #            resolved out-edges are deleted and every edge re-resolved.
    #   hot      unchanged files that name (directly, via an import alias or
    #            the fuzzy key) a symbol whose candidate set changed -- a new
    #            or removed definition of that name: their edges to that name
    #            are deleted and re-resolved (so an unchanged caller re-links
    #            when a same-named symbol appears or disappears).
    #   relink   unchanged files that name a symbol of a changed file whose
    #            candidate set did NOT change (the common "edit a body" case):
    #            only edges that resolve into the changed file are written
    #            again (the old ones went with the DETACH DELETE).
    #
    # Callers are found through the RefIndex (name -> files), never by
    # scanning every cached edge. Candidate lists are ordered by (file,
    # qname), so an incremental pass resolves ties exactly like a full one.

    def _resolve_scope(self, live, cache, changed_set: set[str], deleted: list[str],
                       incremental: bool, prev_files: set[str],
                       before_cands: dict[str, frozenset], incoming: dict | None = None) -> dict:
        from docgraph.live import _mod_tail
        from docgraph.resolve import _norm as _rnorm
        from docgraph.resolve import _stem as _rstem
        symtab = live.symtab
        mi = live.ensure_mod_index()
        imports = live.imports
        work: dict[str, str] = {}
        hot_changed: set[str] = set()
        hot_same: set[str] = set()
        force_files: set[str] = set()
        if not incremental or not live.resolve_ready:
            live.ensure_resolve()
            imports = live.imports
            if incremental:
                # First incremental after (re)loading: the records were built
                # from the post-change cache, so only the scoping below differs.
                pass
        else:
            for rel in changed_set:
                edges = (cache.get(rel) or {}).get("edges") or []
                imports.compute(rel, edges, mi)
                live.refs.add(rel, edges)
        if not incremental:
            for rel in cache:
                work[rel] = "normal"
        else:
            for rel in changed_set:
                work[rel] = "normal"
            now_files = set(symtab.file_index)
            added = now_files - prev_files
            removed = prev_files - now_files
            if added or removed:
                # imports of other files may resolve differently now
                tails: set[str] = set()
                for f in added | removed:
                    tails.add(_rstem(f))
                    base = f.rsplit("/", 1)[-1]
                    tails.add(base.rsplit(".", 1)[0])
                    tails.add(_mod_tail(f))
                cand: set[str] = set()
                for t in tails:
                    cand |= imports.tails.get(t, set())
                cand -= changed_set
                cand -= set(deleted)
                for rel in cand:
                    if rel in cache:
                        imports.compute(rel, (cache.get(rel) or {}).get("edges") or [], mi)
                        force_files.add(rel)
                        work[rel] = "force"
            # names whose candidate set changed vs. names only re-created
            q = symtab.id_qname
            names_now: set[str] = set()
            for rel in changed_set:
                for _lab, _i, name, _qn in symtab.by_file.get(rel, ()):
                    names_now.add(name)
            for name in set(before_cands) | names_now:
                after = frozenset((c[2], q.get(c[1], ""), c[0]) for c in symtab.name_index.get(name, ()))
                before = before_cands.get(name)
                if before is None:
                    before = frozenset()
                if before != after:
                    hot_changed.add(name)
                else:
                    hot_same.add(name)
            refs = live.refs
            for rel in refs.files_for(hot_changed):
                if rel not in work and rel in cache:
                    work[rel] = "hot"
            if incoming is None:
                # no harvested incoming edges (cold live state): re-resolve
                # the callers of re-created names and the importers instead
                for rel in refs.files_for(hot_same):
                    if rel not in work and rel in cache:
                        work[rel] = "relink"
                for rel in changed_set:
                    for f in imports.importers.get(rel, ()):
                        if f not in work and f in cache:
                            work[f] = "imports"
        hot_changed_norm = {_rnorm(n) for n in hot_changed if len(n) >= 4}
        hot_same_norm = {_rnorm(n) for n in hot_same if len(n) >= 4}

        # old resolved edges of force / hot files
        deletes_pairs: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        deletes_all: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        old_inherits: set[tuple[int, int]] = set()
        fw_existing: dict = {}
        unchanged_srcs = [rel for rel, m in work.items() if m != "normal"]
        if unchanged_srcs:
            fw_existing = self.db.framework_nodes_in(unchanged_srcs)
        fw_ids_by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for (f, _line, _name), (lab, nid) in fw_existing.items():
            fw_ids_by_file[f][lab].append(nid)
        for mode in ("force", "hot"):
            files = [rel for rel, m in work.items() if m == mode]
            if not files:
                continue
            ids = symtab.ids_of_files(files)
            for f in files:
                for lab, lst in fw_ids_by_file.get(f, {}).items():
                    ids[lab].extend(lst)
            outs = self.db.resolved_out_edges(ids)
            if mode == "force":
                for rel, fl, tl, a, b, _n in outs:
                    if rel == "INHERITS":
                        old_inherits.add((a, b))
                for rel, fl, tl in self.db.RESOLVED_OUT:
                    if ids.get(fl):
                        deletes_all[(rel, fl, tl)].extend(ids[fl])
                deletes_all[("IMPORTS", "File", "File")].extend(ids.get("File", []))
                deletes_all[("IMPORTS", "File", "Module")].extend(ids.get("File", []))
            else:
                for rel, fl, tl, a, b, n in outs:
                    if n in hot_changed or (len(n) >= 4 and _rnorm(n) in hot_changed_norm):
                        deletes_pairs[(rel, fl, tl)].append((a, b))
                        if rel == "INHERITS":
                            old_inherits.add((a, b))

        resolver = SymbolResolver(
            symtab.name_index, imports.file_imports,
            import_map=imports.import_map, recv_map=imports.recv_map,
            aliases=imports.aliases, method_ids=symtab.method_ids,
            external=imports.external,
        )
        resolver._norm_idx = symtab.norm_index        # maintained by the symbol table
        resolver._undirected = imports.undirected     # maintained by the import index
        resolver._id_names = symtab.id_names

        qname_index = symtab.qname_index
        file_index = symtab.file_index
        rows: dict[str, list[dict]] = defaultdict(list)
        contains_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        modules = live.ensure_modules(self.db)
        new_modules: list[dict] = []
        route_rows: list[dict] = []
        tool_rows: list[dict] = []
        n_edges_seen = 0

        def resolve(name, src_file, prefer_kind=None, raw_extra=None):
            ex = raw_extra or {}
            return resolver.resolve(name, src_file, prefer_kind=prefer_kind,
                                    recv=ex.get("recv"), attr=bool(ex.get("attr")))

        # CONTAINS for changed files
        for rel in changed_set:
            fid = file_index.get(rel)
            if fid is None:
                continue
            for _lab, _i, _name, qname in symtab.by_file.get(rel, ()):
                cur = qname_index.get(qname)
                if cur is None:
                    continue
                label, eid = cur
                parts = qname.split("::")
                if len(parts) == 2:
                    contains_groups[("File", label)].append({"from_id": fid, "to_id": eid})
                elif len(parts) >= 3:
                    parent = qname_index.get("::".join(parts[:-1]))
                    if parent is not None and parent[0] == "Class":
                        contains_groups[("Class", label)].append({"from_id": parent[1], "to_id": eid})

        for rel, mode in sorted(work.items()):
            file_data = cache.get(rel) or {}
            edges = file_data.get("edges") or []
            al = imports.aliases.get(rel) or {}
            force_write = mode in ("force", "hot")

            def needs(target_file: str | None, _mode=mode, _rel=rel) -> bool:
                if _mode in ("force", "hot"):
                    return True
                if _mode == "normal":
                    return _rel in changed_set or (target_file is not None and target_file in changed_set)
                return target_file is not None and target_file in changed_set

            for raw in edges:
                kind = raw["kind"]
                target_name = raw.get("target_name")
                if mode == "imports" and kind != "IMPORTS":
                    continue
                if mode in ("hot", "relink") and kind == "IMPORTS":
                    edge_force = False        # re-link into changed files only
                elif mode in ("hot", "relink"):
                    t = target_name or ""
                    o = al.get(t)
                    names = (t, o) if o else (t,)
                    in_changed = any(x in hot_changed or (len(x) >= 4 and _rnorm(x) in hot_changed_norm)
                                     for x in names)
                    in_same = any(x in hot_same or (len(x) >= 4 and _rnorm(x) in hot_same_norm)
                                  for x in names)
                    if mode == "hot":
                        if not in_changed:
                            if not in_same:
                                continue
                            edge_force = False
                        else:
                            edge_force = True
                    else:
                        if not in_same and not in_changed:
                            continue
                        edge_force = False
                else:
                    edge_force = force_write
                n_edges_seen += 1
                src_qname = raw.get("src_qname")
                line = raw.get("line", 0)
                rex = raw.get("extra") or {}

                def ok(target_file: str | None, _f=edge_force, _m=mode) -> bool:
                    if _f:
                        return True
                    if _m in ("hot", "relink", "imports"):
                        return target_file is not None and target_file in changed_set
                    return needs(target_file)

                if kind == "IMPORTS":
                    src_fid = file_index.get(rel)
                    if src_fid is None:
                        continue
                    matched = [f for f in mi.resolve(target_name or "", rel) if f != rel and f in file_index]
                    if matched:
                        for mpath in matched[:2]:
                            if ok(mpath):
                                rows["IMPORTS_FILE"].append({"from_id": src_fid, "to_id": file_index[mpath]})
                    else:
                        if not (edge_force or rel in changed_set):
                            continue
                        if target_name not in modules:
                            mid = self._new_id()
                            modules[target_name] = {"id": mid, "name": target_name,
                                                    "language": file_data.get("language", "")}
                            new_modules.append(modules[target_name])
                        rows["IMPORTS_MODULE"].append({"from_id": src_fid, "to_id": modules[target_name]["id"]})
                    continue

                if kind == "IMPORTS_SYMBOL":
                    src_fid = file_index.get(rel)
                    if src_fid is None or not target_name:
                        continue
                    res = resolve(target_name, rel)
                    if res.target is None:
                        continue
                    tlabel, tid, tfile = res.target
                    if ok(tfile):
                        if tlabel == "Class":
                            rows["IMPORTS_SYMBOL_CLASS"].append({"from_id": src_fid, "to_id": tid})
                        elif tlabel == "Function":
                            rows["IMPORTS_SYMBOL_FUNC"].append({"from_id": src_fid, "to_id": tid})
                    continue

                if kind in ("ROUTE", "TOOL"):
                    hid = hfile = None
                    if src_qname and src_qname in qname_index and qname_index[src_qname][0] == "Function":
                        hid, hfile = qname_index[src_qname][1], rel
                    elif target_name:
                        res = resolve(target_name, rel, prefer_kind="Function")
                        if res.target is not None and res.target[0] == "Function":
                            hid, hfile = res.target[1], res.target[2]
                    label = "Route" if kind == "ROUTE" else "Tool"
                    name = rex.get("name") or target_name or ""
                    if rel in changed_set:
                        nid = self._new_id()
                        if label == "Route":
                            route_rows.append({
                                "id": nid, "name": name, "method": rex.get("method") or "",
                                "path": rex.get("path") or "", "framework": rex.get("framework") or "",
                                "file": rel, "line": int(line or 0),
                                "handler": src_qname or target_name or "",
                            })
                        else:
                            tool_rows.append({
                                "id": nid, "name": name, "kind": rex.get("tool_kind") or "tool",
                                "framework": rex.get("framework") or "mcp",
                                "file": rel, "line": int(line or 0),
                                "handler": src_qname or target_name or "",
                            })
                    else:
                        found = fw_existing.get((rel, int(line or 0), name))
                        nid = found[1] if found else None
                        if nid is None or not ok(hfile):
                            continue
                    if hid is not None and nid is not None:
                        rows["HANDLES_ROUTE" if label == "Route" else "HANDLES_TOOL"].append(
                            {"from_id": nid, "to_id": hid})
                    continue

                if not src_qname or src_qname not in qname_index:
                    continue
                src_label, src_id = qname_index[src_qname]

                if kind == "CALLS":
                    if src_label != "Function":
                        continue
                    res = resolve(target_name, rel, prefer_kind="Function", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Function":
                        if ok(res.target[2]):
                            rows["CALLS"].append({"from_id": src_id, "to_id": res.target[1], "line": line,
                                                  "confidence": float(res.confidence), "method": res.method})
                        alts = [c for c in res.candidates if c[0] == "Function" and c[1] != res.target[1]]
                        for c in alts[:4]:
                            if ok(c[2]):
                                rows["CALLS_CANDIDATE"].append({"from_id": src_id, "to_id": c[1], "line": line,
                                                                "confidence": candidate_confidence(len(alts) + 1),
                                                                "method": "alternative"})
                    elif res.candidates:
                        cands = [c for c in res.candidates if c[0] == "Function"]
                        for c in cands[:5]:
                            if ok(c[2]):
                                rows["CALLS_CANDIDATE"].append({"from_id": src_id, "to_id": c[1], "line": line,
                                                                "confidence": candidate_confidence(len(cands)),
                                                                "method": res.method})
                elif kind == "INSTANTIATES":
                    if src_label != "Function":
                        continue
                    res = resolve(target_name, rel, prefer_kind="Class", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Class" and ok(res.target[2]):
                        rows["INSTANTIATES"].append({"from_id": src_id, "to_id": res.target[1], "line": line,
                                                     "confidence": float(res.confidence), "method": res.method})
                elif kind == "INHERITS":
                    if src_label != "Class":
                        continue
                    res = resolve(target_name, rel, prefer_kind="Class", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Class" and ok(res.target[2]):
                        rows["INHERITS"].append({"from_id": src_id, "to_id": res.target[1],
                                                 "confidence": float(res.confidence), "method": res.method})
                elif kind == "DECORATED_BY":
                    res = resolve(target_name, rel, prefer_kind="Function", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Function" and ok(res.target[2]):
                        if src_label == "Function":
                            rows["DECORATED_FUNC"].append({"from_id": src_id, "to_id": res.target[1]})
                        elif src_label == "Class":
                            rows["DECORATED_CLASS"].append({"from_id": src_id, "to_id": res.target[1]})

        # OVERRIDES are recomputed (after the INHERITS writes) for every class
        # whose ancestry can have changed: classes of changed files, sources
        # of INHERITS rows written or deleted here, and their descendants.
        if incremental:
            touched_cls: set[int] | None = {r["from_id"] for r in rows.get("INHERITS", [])}
            touched_cls |= {a for a, _b in old_inherits}
            for rel in changed_set:
                for lab, i, _n, _q in symtab.by_file.get(rel, ()):
                    if lab == "Class":
                        touched_cls.add(int(i))
        else:
            touched_cls = None
        # Re-point the harvested incoming edges of unchanged callers whose
        # target name kept exactly its candidate set: the resolution of such
        # an edge cannot change (same candidates, same caller), only the
        # target's id did -- O(incoming edges), no resolver call.
        n_remap = 0
        if incremental and incoming:
            key_of = {"CALLS": "CALLS", "CALLS_CANDIDATE": "CALLS_CANDIDATE", "INSTANTIATES": "INSTANTIATES",
                      "INHERITS": "INHERITS", "IMPORTS": "IMPORTS_FILE"}
            q_old = incoming["keys"]
            for rel, fl, tl, a, af, b, props in incoming["edges"]:
                if af in work or af in changed_set or af not in cache:
                    continue
                tgt = q_old.get(b)
                if tgt is None:
                    continue
                name, qn = tgt
                if tl == "File":
                    nb = file_index.get(qn)
                else:
                    if name not in hot_same:
                        continue
                    cur = qname_index.get(qn)
                    nb = cur[1] if cur is not None and cur[0] == tl else None
                if nb is None:
                    continue
                if rel == "DECORATED_BY":
                    key = "DECORATED_FUNC" if fl == "Function" else "DECORATED_CLASS"
                elif rel == "IMPORTS_SYMBOL":
                    key = "IMPORTS_SYMBOL_CLASS" if tl == "Class" else "IMPORTS_SYMBOL_FUNC"
                elif rel == "HANDLES":
                    key = "HANDLES_ROUTE" if fl == "Route" else "HANDLES_TOOL"
                else:
                    key = key_of[rel]
                row = {"from_id": a, "to_id": nb}
                row.update(props)
                rows[key].append(row)
                if rel == "INHERITS" and touched_cls is not None:
                    touched_cls.add(a)
                n_remap += 1
        modes = defaultdict(int)
        for m in work.values():
            modes[m] += 1
        return {
            "rows": rows, "contains": contains_groups, "routes": route_rows, "tools": tool_rows,
            "modules": new_modules, "deletes_pairs": deletes_pairs, "deletes_all": deletes_all,
            "overrides_for": touched_cls,
            "stats": dict(resolver.stats),
            "scope": {"scoped": bool(incremental), "files": dict(modes), "edges_resolved": n_edges_seen,
                      "edges_remapped": n_remap,
                      "hot_changed": len(hot_changed), "hot_same": len(hot_same)},
        }

    def _refresh_overrides(self, touched: set[int] | None, symtab) -> None:
        """(child method -> ancestor method) pairs of the same name over the
        inheritance closure. touched=None: every class (full pass). Else the
        touched classes and their descendants: their old OVERRIDES rows are
        deleted and rewritten, so the result equals a full recompute."""
        pairs = self.db.inherits_pairs()
        if not pairs and not touched:
            return
        parents: dict[int, set[int]] = defaultdict(set)
        children: dict[int, set[int]] = defaultdict(set)
        for a, b in pairs:
            parents[int(a)].add(int(b))
            children[int(b)].add(int(a))
        if touched is None:
            classes = set(parents)
        else:
            classes = set()
            stack = list(touched)
            while stack:
                c = stack.pop()
                if c in classes:
                    continue
                classes.add(c)
                stack.extend(children.get(c, ()))
        meth_cache: dict[int, dict[str, int]] = {}

        def methods_of(cid: int) -> dict[str, int]:
            hit = meth_cache.get(cid)
            if hit is not None:
                return hit
            out: dict[str, int] = {}
            q = symtab.id_qname.get(cid)
            if q:
                f = q.split("::", 1)[0]
                depth = q.count("::") + 1
                pre = q + "::"
                for lab, i, name, qn in symtab.by_file.get(f, ()):
                    if lab == "Function" and qn.startswith(pre) and qn.count("::") == depth:
                        out.setdefault(name, int(i))
            meth_cache[cid] = out
            return out

        rows: list[dict] = []
        seen: set[tuple[int, int]] = set()
        old_srcs: list[int] = []
        for child in sorted(classes):
            cm = methods_of(child)
            old_srcs.extend(cm.values())
            anc: set[int] = set()
            stack = list(parents.get(child, ()))
            while stack:
                p = stack.pop()
                if p in anc:
                    continue
                anc.add(p)
                stack.extend(parents.get(p, ()))
            for p in sorted(anc):
                pm = methods_of(p)
                for name, mid in cm.items():
                    pid = pm.get(name)
                    if pid is None or pid == mid or (mid, pid) in seen:
                        continue
                    seen.add((mid, pid))
                    rows.append({"from_id": mid, "to_id": pid})
        if touched is not None and old_srcs:
            self.db.delete_out_edges("OVERRIDES", "Function", "Function", old_srcs)
        if rows:
            self.db.insert_edges("OVERRIDES", "Function", "Function", rows, validate=False)

    def _write_resolved(self, res: dict) -> None:
        db = self.db
        for (rel, fl, tl), ids in res["deletes_all"].items():
            db.delete_out_edges(rel, fl, tl, sorted(set(ids)))
        for (rel, fl, tl), pairs in res["deletes_pairs"].items():
            db.delete_edge_pairs(rel, fl, tl, sorted(set(pairs)))
        if res["modules"]:
            db.insert_nodes("Module", res["modules"])
        if res["routes"]:
            db.insert_nodes("Route", res["routes"])
        if res["tools"]:
            db.insert_nodes("Tool", res["tools"])
        rows = res["rows"]
        n_edges = sum(len(r) for r in res["contains"].values()) + sum(len(r) for r in rows.values())
        if not n_edges:
            if res.get("overrides_for"):
                self._refresh_overrides(res["overrides_for"], self.live.symtab)
            return
        spec = (("CALLS", "CALLS", "Function", "Function"), ("INSTANTIATES", "INSTANTIATES", "Function", "Class"),
                ("INHERITS", "INHERITS", "Class", "Class"), ("DECORATED_FUNC", "DECORATED_BY", "Function", "Function"),
                ("DECORATED_CLASS", "DECORATED_BY", "Class", "Function"), ("IMPORTS_FILE", "IMPORTS", "File", "File"),
                ("IMPORTS_MODULE", "IMPORTS", "File", "Module"), ("IMPORTS_SYMBOL_CLASS", "IMPORTS_SYMBOL", "File", "Class"),
                ("IMPORTS_SYMBOL_FUNC", "IMPORTS_SYMBOL", "File", "Function"),
                ("CALLS_CANDIDATE", "CALLS_CANDIDATE", "Function", "Function"),
                ("HANDLES_ROUTE", "HANDLES", "Route", "Function"), ("HANDLES_TOOL", "HANDLES", "Tool", "Function"))
        with _bar() as prog:
            task = prog.add_task("Writing graph edges", total=n_edges)
            cb = lambda n: prog.advance(task, n)  # noqa: E731
            for (fl, tl), r in res["contains"].items():
                db.insert_edges("CONTAINS", fl, tl, r, on_progress=cb, validate=False)
            for key, rel, fl, tl in spec:
                r = rows.get(key)
                if r:
                    db.insert_edges(rel, fl, tl, r, on_progress=cb, validate=False)
        if res.get("overrides_for") is None or res["overrides_for"]:
            self._refresh_overrides(res.get("overrides_for"), self.live.symtab)

    # ---- incremental analytics (id-based, batched) -------------------------

    def _harvest_layout_ids(self, old_ids: dict[str, list[int]], files: list[str]) -> dict:
        """Before the delete step: positions / ranks / communities of the
        nodes about to be recreated, keyed by stable names (qname / path),
        read by primary key."""
        out: dict = {"by_q": {}, "by_p": {}, "file_comm": {}}
        if not files:
            return out
        symtab = self.live.symtab
        q_of = symtab.id_qname
        path_of = {symtab.file_index[f]: f for f in files if f in symtab.file_index}
        try:
            for label in ("Function", "Class", "Variable"):
                for i, (x, y, pr) in self.db.layout_by_ids(label, old_ids.get(label, [])).items():
                    q = q_of.get(i)
                    if q:
                        out["by_q"][q] = (x, y, pr)
            for i, (x, y, _pr) in self.db.layout_by_ids("File", old_ids.get("File", [])).items():
                p = path_of.get(i)
                if p:
                    out["by_p"][p] = (x, y)
        except Exception:
            log.debug("layout harvest failed", exc_info=True)
        try:
            for i, c in self.db.communities_of(old_ids.get("File", [])).items():
                p = path_of.get(i)
                if p:
                    out["file_comm"][p] = c
        except Exception:
            log.debug("community harvest failed", exc_info=True)
        return out

    def _new_nodes(self, changed: set[str]) -> dict[str, list]:
        """[(label, id, name, qname, file)] of the nodes of changed files + File rows."""
        symtab = self.live.symtab
        out: list[tuple[str, int, str, str, str]] = []
        for f in sorted(changed):
            fid = symtab.file_index.get(f)
            if fid is not None:
                out.append(("File", int(fid), f, f, f))
            for lab, i, name, qn in symtab.by_file.get(f, ()):
                out.append((lab, int(i), name, qn, f))
        return out

    def _incremental_layout(self, changed: set[str], harvest: dict, nodes: list) -> dict[int, tuple]:
        """Positions of the new nodes: a changed file keeps its previous
        centre (a new file is placed next to its neighbours / community) and
        its symbols re-spiral around it in line order. Nothing else moves.
        One batched write per label."""
        import numpy as np
        from docgraph import layout as _layout
        by_p = harvest.get("by_p") or {}
        by_label: dict[str, list[int]] = defaultdict(list)
        for lab, i, _n, _q, _f in nodes:
            if lab != "File":
                by_label[lab].append(i)
        lines = self.db.node_lines(by_label) if by_label else {}
        per_file: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        file_id: dict[str, int] = {}
        for lab, i, _n, _q, f in nodes:
            if lab == "File":
                file_id[f] = i
            else:
                per_file[f].append((lab, i, lines.get(i, 0)))
        extent = None
        pos: dict[int, tuple[float, float]] = {}
        for f in sorted(changed):
            fid = file_id.get(f)
            if fid is None:
                continue
            center = by_p.get(f)
            if center is None:
                if extent is None:
                    extent = self.db.layout_extent()
                ids = [i for _lab, i, _l in per_file.get(f, ())] + [fid]
                anchor = self.db.neighbor_center(ids)
                if anchor is None:
                    anchor = self.db.community_center_of(fid)
                center = _layout.place_file(anchor, f, extent)
            pos[fid] = (float(center[0]), float(center[1]))
            syms = sorted(per_file.get(f, ()), key=lambda t: (t[2], t[1]))
            off = _layout.spiral_offsets(len(syms))
            for (lab, i, _l), (dx, dy) in zip(syms, off.tolist()):
                pos[i] = (center[0] + dx, center[1] + dy)
        label_of = {i: lab for lab, i, _n, _q, _f in nodes}
        grouped: dict[str, list[int]] = defaultdict(list)
        for i in pos:
            grouped[label_of[i]].append(i)
        for lab, ids in grouped.items():
            self.db.set_node_values(lab, np.array(ids, np.int64),
                                    {"x": np.array([pos[i][0] for i in ids]),
                                     "y": np.array([pos[i][1] for i in ids])})
        return pos

    def _incremental_pagerank(self, harvest: dict, nodes: list) -> dict[int, float]:
        """Big-graph incremental: nodes of changed files take the rank their
        qname had before (0 for new symbols); their files re-sum."""
        import numpy as np
        by_q = harvest.get("by_q") or {}
        pr: dict[int, float] = {}
        per_file: dict[str, float] = defaultdict(float)
        grouped: dict[str, list[int]] = defaultdict(list)
        for lab, i, _n, q, f in nodes:
            if lab in ("Function", "Class"):
                v = float((by_q.get(q) or (0, 0, 0.0))[2])
                pr[i] = v
                per_file[f] += v
                grouped[lab].append(i)
            elif lab == "Variable":
                pr[i] = 0.0
        for lab, i, _n, _q, f in nodes:
            if lab == "File":
                pr[i] = per_file.get(f, 0.0)
                grouped["File"].append(i)
        for lab, ids in grouped.items():
            self.db.set_node_values(lab, np.array(ids, np.int64), {"pagerank": np.array([pr[i] for i in ids])})
        return pr

    def _incremental_communities(self, harvest: dict, nodes: list) -> dict[int, int]:
        """Big-graph incremental: nodes of changed files join their file's
        previous community, or the majority community of their neighbours."""
        from collections import Counter as _Counter
        if not nodes or not self.db.has_table("Community"):
            return {}
        file_comm = dict(harvest.get("file_comm") or {})
        unknown = [i for lab, i, _n, _q, f in nodes if f not in file_comm]
        if unknown:
            votes: dict[str, _Counter] = defaultdict(_Counter)
            f_of = {i: f for _lab, i, _n, _q, f in nodes}
            for a, c in self.db.neighbor_communities(unknown):
                votes[f_of.get(a, "")][c] += 1
            for f, cnt in votes.items():
                if f and f not in file_comm and cnt:
                    file_comm[f] = cnt.most_common(1)[0][0]
        members: dict[str, list[dict]] = defaultdict(list)
        comm: dict[int, int] = {}
        for lab, i, _n, _q, f in nodes:
            c = file_comm.get(f)
            if c is not None and lab != "Variable":
                members[lab].append({"from_id": i, "to_id": int(c)})
                comm[i] = int(c)
        for lab, rows in members.items():
            self.db.insert_edges("MEMBER_OF", lab, "Community", rows, validate=False)
        return comm

    def _patch_tiles(self, state: dict, nodes: list, pos: dict, pr: dict, comm: dict,
                     removed_ids: dict, tests: set[int]) -> bool:
        """Apply this pass to the in-memory tile sidecar (loaded once) and
        publish the new generation. False when there is nothing to patch
        (no sidecar yet / an old format): the caller rebuilds."""
        import numpy as np
        from docgraph import tiles as _tiles
        live = self.live
        journal = self.db.journal
        if journal is None or self._tiles_full:
            return False
        if self.tile_patcher is None:
            if live.tiles is None:
                loaded = _tiles.load(self.cfg.data_dir / "tiles")
                if loaded is None:
                    return False
                live.tiles = loaded
            man, arrays = live.tiles
            if "sym_file" not in arrays:
                return False
        elif live.tiles is None and not (self.cfg.data_dir / "tiles" / "manifest.json").exists():
            return False
        gone = []
        for lab in ("File", "Class", "Function", "Variable"):
            gone.extend(removed_ids.get(lab, []))
        kind_of = {"File": 0, "Class": 1, "Function": 2, "Variable": 3}
        tn = [n for n in nodes if n[1] in pos]
        fid_of = self.live.symtab.file_index
        delta = _tiles.TileDelta(
            removed=np.asarray(gone, np.int64),
            ids=np.array([n[1] for n in tn], np.int64),
            kinds=np.array([kind_of[n[0]] for n in tn], np.uint8),
            x=np.array([pos[n[1]][0] for n in tn]), y=np.array([pos[n[1]][1] for n in tn]),
            pagerank=np.array([pr.get(n[1], 0.0) for n in tn]),
            file_id=np.array([fid_of.get(n[4], -1) for n in tn], np.int64),
            community=np.array([comm.get(n[1], -1) for n in tn], np.int64),
            names=[n[2] for n in tn],
            flags=np.array([_tiles.FLAG_TEST if n[1] in tests else 0 for n in tn], np.uint8),
            ops=journal.ops)
        gen = int(state.get("tiles_generation", 0) or 0) + 1
        if self.tile_patcher is not None:
            self.tile_patcher(delta, gen)
            state["tiles_generation"] = gen
            state["tiles"] = {"generation": gen, "patched": True, "deferred": True}
            return True
        new_arrays, new_man = _tiles.patch(arrays, man, delta, gen)
        live.tiles = (new_man, new_arrays)
        state["tiles_generation"] = gen
        state["tiles"] = {"generation": gen, "counts": new_man.get("counts"),
                          "seconds": new_man.get("build_seconds"), "patched": True}
        self._publish_tiles(new_man, new_arrays)
        return True

    def _publish_tiles(self, man: dict, arrays: dict) -> None:
        """Hand a generation to the host (served from memory at once) and
        persist it -- in the background when the host provides a sink."""
        from docgraph import tiles as _tiles
        sink = self.tile_sink
        if sink is not None:
            try:
                sink(man, arrays)
                return
            except Exception:
                log.warning("tile sink failed; persisting inline", exc_info=True)
        _tiles.write(self.cfg.data_dir / "tiles", arrays, man)

    # ---- History helpers ----
    def _owner(self, rel: str) -> tuple[Path, str] | None:
        for root, prefix in self.cfg.roots_with_prefix():
            if prefix == "":
                return root, rel
            if rel.startswith(prefix):
                return root, rel[len(prefix):]
        return None

    def _git_roots(self) -> set[Path]:
        cached = getattr(self.live, "git_roots", None)
        if cached is not None:
            return cached
        if not hasattr(self, "_git_roots_cache"):
            roots = set()
            for root, _p in self.cfg.roots_with_prefix():
                if root == self.cfg.external_dir:
                    continue
                if _history.is_git_repo(root):
                    roots.add(root)
            self._git_roots_cache = roots
            self.live.git_roots = roots
        return self._git_roots_cache

    def _blame_jobs(self, rels: list[str]) -> list[tuple[str, Path, str]]:
        git_roots = self._git_roots()
        jobs = []
        for rel in rels:
            own = self._owner(rel)
            if own and own[0] in git_roots:
                jobs.append((rel, own[0], own[1]))
        return jobs

    def _primary_head(self) -> str | None:
        if self.cfg.repo_root in self._git_roots():
            return _history.head(self.cfg.repo_root)
        return None

    @staticmethod
    def _history_cols(blame: list | None, s: int, e: int) -> dict:
        h = _history.symbol_span_history(blame or [], s, e) if blame else {}
        return {
            "first_seen_commit": h.get("fc", ""), "first_seen_ts": int(h.get("fts", 0) or 0),
            "last_changed_commit": h.get("lc", ""), "last_changed_ts": int(h.get("lts", 0) or 0),
        }

    def _refresh_pending_history(self, state: dict, changed: set[str]) -> bool:
        """Re-blame files that had uncommitted lines at their last blame, if
        HEAD moved since. Updates the history columns in place."""
        pending = [f for f in (state.get("history_pending") or []) if f not in changed]
        if not pending or not getattr(self.cfg, "history", True):
            return False
        head_now = self._primary_head() or ""
        if head_now and head_now == state.get("history_head"):
            return False
        try:
            blames = _history.blame_many(self._blame_jobs(pending), workers=4)
            spans = self.db.entity_spans(list(blames.keys()))
        except Exception:
            log.debug("history refresh failed", exc_info=True)
            return False
        by_label: dict[str, list[dict]] = defaultdict(list)
        still: set[str] = set()
        for sp in spans:
            b = blames.get(sp["file"])
            if not b:
                continue
            h = _history.symbol_span_history(b, int(sp["s"] or 1), int(sp["e"] or 1))
            if not h:
                continue
            if h.get("lc") == _history.UNCOMMITTED:
                still.add(sp["file"])
            by_label[sp["label"]].append({"id": sp["id"], "fc": h["fc"], "fts": int(h["fts"]),
                                          "lc": h["lc"], "lts": int(h["lts"])})
        for label, rows in by_label.items():
            try:
                self.db.set_history(label, rows)
            except Exception:
                log.debug("set_history failed", exc_info=True)
        state["history_pending"] = sorted(
            (set(state.get("history_pending") or []) - set(pending)) | still
            | {f for f in pending if f not in blames})
        state["history_head"] = head_now
        return True

    # ---- Graph analytics: PageRank, communities, layout, tiles ----
    #
    # One global pass reads every node and edge ONCE as numpy arrays (Arrow)
    # and derives PageRank (sparse power iteration), communities (Louvain on
    # a folded graph when big), the world layout and the LOD tile sidecar.
    #
    # Small graphs (<= full_recompute_max_nodes) recompute PageRank and
    # communities on every dirty pass, as before. Bigger graphs patch them
    # for the changed files only (new nodes inherit their previous rank /
    # their file's community) until the files changed since the last global
    # pass exceed recompute_drift of all files. The layout is incremental for
    # every size: surviving nodes keep their stored position, changed files
    # re-spiral their symbols around the file's previous centre, and a full
    # relayout happens only on a full reindex or once the drift threshold is
    # crossed -- so tiles outside the changed files stay byte-identical.

    ANALYTICS_VERSION = 1

    def _graph_analytics(self, full: bool, changed: set[str], deleted: set[str],
                         n_files: int, state: dict, harvest: dict,
                         emit: Callable[[str], None] | None = None,
                         removed_ids: dict | None = None, chunk_ids=None) -> dict:
        t0 = time.perf_counter()
        symtab = self.live.symtab
        n_nodes = len(symtab.id_qname) + len(symtab.file_index)
        big = n_nodes > int(getattr(self.cfg, "full_recompute_max_nodes", 50_000) or 0)
        drift = int(state.get("analytics_drift", 0) or 0) + len(changed) + len(deleted)
        limit = max(1.0, float(getattr(self.cfg, "recompute_drift", 0.05) or 0.0) * max(1, n_files))
        stale = state.get("analytics_version") != self.ANALYTICS_VERSION
        over = drift > limit
        stats = full or stale or over or not big
        want_tiles = bool(getattr(self.cfg, "tiles", True))
        if want_tiles and (full or not state.get("layout_complete")):
            missing_layout = self.db.missing_positions() > len(changed)
        else:
            missing_layout = False
        relayout = want_tiles and (full or stale or over or missing_layout)
        deferred = False
        if (stats or relayout) and not full and big and self.defer_global:
            # The global pass (PageRank, communities, layout: tens of seconds
            # on a big graph) goes to the host's background maintenance;
            # this pass patches like any other and the UI shows "refreshing".
            deferred = True
            stats = relayout = False
        info: dict = {"nodes": n_nodes, "big": big, "stats": "global" if stats else "patched",
                      "layout": ("global" if relayout else "incremental") if want_tiles else "off",
                      "drift_files": 0 if relayout or (stats and big) else drift,
                      "deferred": deferred}
        nodes = self._new_nodes(changed) if not (stats and relayout) else []
        pos: dict = {}
        pr: dict = {}
        comm: dict = {}
        if want_tiles and not relayout:
            if emit:
                emit("layout")
            pos = self._incremental_layout(changed, harvest, nodes)
        if not stats:
            if emit:
                emit("pagerank")
            pr = self._incremental_pagerank(harvest, nodes)
            if getattr(self.cfg, "communities", True):
                comm = self._incremental_communities(harvest, nodes)
        patched = False
        if want_tiles and not stats and not relayout:
            if emit:
                emit("tiles")
            try:
                patched = self._patch_tiles(state, nodes, pos, pr, comm, removed_ids or {},
                                            self._new_test_ids)
            except Exception:
                log.warning("tile patch failed; rebuilding", exc_info=True)
                patched = False
        n_comm = None
        if not patched:
            n_comm = self._global_pass(stats=stats, relayout=relayout, tiles=want_tiles,
                                       state=state, emit=emit)
        if n_comm is not None:
            info["communities"] = n_comm
        if relayout:
            state["layout_complete"] = True
        if relayout or (stats and big) or (stats and not want_tiles):
            state["analytics_drift"] = 0
        else:
            state["analytics_drift"] = drift
        state["analytics_pending"] = bool(deferred or state.get("analytics_pending")) and not (stats and big)
        state["analytics_version"] = self.ANALYTICS_VERSION
        info["tiles"] = "patched" if patched else ("rebuilt" if want_tiles else "off")
        info["seconds"] = round(time.perf_counter() - t0, 3)
        state["analytics"] = info
        _console.print(f"[cyan]Analytics[/]: stats {info['stats']}, layout {info['layout']}, "
                       f"tiles {info['tiles']}{' (global pass deferred)' if deferred else ''} "
                       f"({info['seconds']:.2f}s)")
        return info

    def run_maintenance(self) -> dict:
        """Background work an incremental pass handed off (host only): the
        global analytics pass when drift / staleness asked for it, and
        SIMILAR_TO for big batches of new entities. Runs as its own writer
        job after the index response has gone out."""
        state = self._load_state()
        done: dict = {}
        t0 = time.perf_counter()
        pend = state.get("similar_pending") or {}
        if pend:
            symtab = self.live.ensure_symtab(self.db)
            for label, ids in pend.items():
                ids = [int(i) for i in ids if int(i) in symtab.id_qname]
                if ids:
                    self._similar_for_ids(label, ids, state, inline=True)
                    done[f"similar_{label}"] = len(ids)
            state["similar_pending"] = {}
        if state.get("analytics_pending"):
            want_tiles = bool(getattr(self.cfg, "tiles", True))
            n_comm = self._global_pass(stats=True, relayout=want_tiles, tiles=want_tiles, state=state)
            if n_comm is not None:
                state["communities"] = n_comm
            state["analytics_drift"] = 0
            state["analytics_pending"] = False
            if want_tiles:
                state["layout_complete"] = True
            done["global"] = True
        done["seconds"] = round(time.perf_counter() - t0, 3)
        state["maintenance"] = dict(done, at=time.time())
        self._save_state(state)
        return done

    def _global_pass(self, stats: bool, relayout: bool, tiles: bool, state: dict,
                     emit: Callable[[str], None] | None = None) -> int | None:
        """Arrays in once; PageRank / communities (stats), layout (relayout),
        writes, then the tile sidecar."""
        plan = self._global_compute(self.db, stats, relayout, tiles, emit)
        if plan is None:
            return None
        return self._global_apply(plan, state, emit)

    def _global_compute(self, db, stats: bool, relayout: bool, tiles: bool,
                        emit: Callable[[str], None] | None = None) -> dict | None:
        """Read + compute half of the global pass -- reads only, so the host
        runs it on the read-only handle outside the writer lock."""
        import numpy as np
        from docgraph import tiles as _tiles
        from docgraph.rank import RANK_RELS, _Graph
        if not (stats or relayout or tiles):
            return None
        src = db.layout_source()
        ids = src["ids"]
        n = len(ids)
        if n == 0:
            z = np.zeros(0, np.int64)
            return {"empty": True, "src": src, "stats": stats, "relayout": relayout, "tiles": tiles,
                    "ra": z, "rb": z, "e_c": np.zeros(0, np.float32), "e_k": np.zeros(0, np.uint8),
                    "comm": None, "comm_of": None, "n_comm": 0 if stats else None}
        kinds = src["kinds"]
        order = np.argsort(ids, kind="stable")
        sid = ids[order]

        def rows_of(q: np.ndarray) -> np.ndarray:
            p = np.searchsorted(sid, q)
            p = np.clip(p, 0, n - 1)
            r = order[p]
            return np.where(sid[p] == q, r, -1)

        e_a, e_b, e_c, e_k = db.edge_endpoints(_tiles.EDGE_KINDS, with_conf=True)
        ra, rb = rows_of(e_a), rows_of(e_b)
        path_row = {src["files"][i]: i for i in np.nonzero(kinds == 0)[0].tolist()}
        file_row = np.array([path_row.get(f, -1) for f in src["files"]], dtype=np.int64)
        pr = src["pr"].astype(np.float64)
        n_comm: int | None = None
        comm_of = None
        comm = None
        if stats:
            if emit:
                emit("pagerank")
            rank_k = [_tiles.EDGE_KIND_ID[r] for r in RANK_RELS]
            m = np.isin(e_k, rank_k)
            g = _Graph(e_a[m], e_b[m])
            scores = g.pagerank()
            pr = np.zeros(n)
            if g.n:
                rr = rows_of(g.nodes)
                ok = rr >= 0
                pr[rr[ok]] = scores[ok]
            pr[kinds == 0] = 0.0
            sym = (kinds != 0) & (file_row >= 0)
            np.add.at(pr, file_row[sym], pr[sym])
            if getattr(self.cfg, "communities", True):
                if emit:
                    emit("communities")
                comm_of, comm = self._communities_from_arrays(src, pr, ra, rb, e_k, e_c, sid, order)
                n_comm = len(comm[0])
        if comm_of is None:
            mn, mc = db.memberships()
            comm_of = np.full(n, -1, dtype=np.int64)
            if len(mn):
                r = rows_of(mn)
                ok = r >= 0
                comm_of[r[ok]] = mc[ok]
        # Files with no graph edges (READMEs, configs, empty __init__.py) have
        # no Louvain community; for layout / tiles they join the dominant
        # community of their directory (MEMBER_OF itself is left as is).
        comm_of = comm_of.copy()
        files_idx = np.nonzero(kinds == 0)[0]
        lone = [int(i) for i in files_idx.tolist() if comm_of[i] < 0]
        if lone and len(lone) < len(files_idx):
            from collections import Counter as _Counter
            by_dir: dict[str, _Counter] = defaultdict(_Counter)
            for i in files_idx.tolist():
                if comm_of[i] >= 0:
                    by_dir[src["files"][i].rsplit("/", 1)[0] if "/" in src["files"][i] else ""][int(comm_of[i])] += 1
            for i in lone:
                f = src["files"][i]
                cnt = by_dir.get(f.rsplit("/", 1)[0] if "/" in f else "")
                if cnt:
                    comm_of[i] = cnt.most_common(1)[0][0]
            sym = (kinds != 0) & (file_row >= 0) & (comm_of < 0)
            comm_of[sym] = comm_of[file_row[sym]]
        x, y = src["x"].astype(np.float64), src["y"].astype(np.float64)
        comm_xy: dict[int, tuple[float, float, float]] = {}
        if relayout:
            if emit:
                emit("layout")
            from docgraph import layout as _layout
            ok = (ra >= 0) & (rb >= 0)
            fa, fb = file_row[ra[ok]], file_row[rb[ok]]
            ok2 = (fa >= 0) & (fb >= 0) & (fa != fb)
            fa, fb = np.minimum(fa[ok2], fb[ok2]), np.maximum(fa[ok2], fb[ok2])
            if len(fa):
                key, inv = np.unique(fa * n + fb, return_inverse=True)
                fw = np.bincount(inv).astype(np.float64)
                fa, fb = key // n, key % n
            else:
                fw = np.zeros(0)
            files_idx = np.nonzero(kinds == 0)[0]
            fcomm = {int(i): int(comm_of[i]) for i in files_idx.tolist() if comm_of[i] >= 0}
            res = _layout.compute_layout(
                kinds, file_row, src["line"], {int(i): src["files"][i] for i in files_idx.tolist()},
                fcomm, fa, fb, fw, pagerank=pr,
                progress=lambda s: (_console.print(f"[dim]{s}[/]"), emit and emit("layout: " + s)))
            x, y = res.x, res.y
            for k_i, key in enumerate(res.cluster_keys):
                if key[0] == "c":
                    comm_xy[int(key[1])] = (float(res.cluster_x[k_i]), float(res.cluster_y[k_i]),
                                            float(res.cluster_r[k_i]))
        return {"empty": False, "src": src, "ids": ids, "kinds": kinds, "stats": stats,
                "relayout": relayout, "tiles": tiles, "ra": ra, "rb": rb, "e_c": e_c, "e_k": e_k,
                "file_row": file_row, "pr": pr, "x": x, "y": y, "comm_xy": comm_xy,
                "comm": comm, "comm_of": comm_of, "n_comm": n_comm}

    def _global_apply(self, plan: dict, state: dict,
                      emit: Callable[[str], None] | None = None) -> int | None:
        """Write half of the global pass (needs the writer)."""
        import numpy as np
        db = self.db
        src = plan["src"]
        stats, relayout, tiles = plan["stats"], plan["relayout"], plan["tiles"]
        n_comm = plan["n_comm"]
        if plan["empty"]:
            if tiles:
                self._write_tiles(src, plan["ra"], plan["rb"], plan["e_c"], plan["e_k"], None, state)
            return n_comm
        if plan["comm"] is not None:
            rows, members = plan["comm"]
            db.replace_communities(rows, members)
            _console.print(f"[cyan]Communities[/]: {len(rows)} detected")
        ids, kinds = plan["ids"], plan["kinds"]
        pr, x, y, comm_xy = plan["pr"], plan["x"], plan["y"], plan["comm_xy"]
        ra, rb, e_c, e_k = plan["ra"], plan["rb"], plan["e_c"], plan["e_k"]
        comm_of, file_row = plan["comm_of"], plan["file_row"]
        # ---- writes ----
        if stats or relayout:
            label_of = {0: "File", 1: "Class", 2: "Function", 3: "Variable"}
            for k, label in label_of.items():
                rows = np.nonzero(kinds == k)[0]
                if not len(rows):
                    continue
                cols: dict = {}
                if stats and label != "Variable":
                    cols["pagerank"] = pr[rows]
                if relayout:
                    cols["x"] = x[rows]
                    cols["y"] = y[rows]
                if cols:
                    db.set_node_values(label, ids[rows], cols)
            if comm_xy:
                cids = np.array(sorted(comm_xy), dtype=np.int64)
                db.set_node_values("Community", cids, {
                    "x": np.array([comm_xy[c][0] for c in cids.tolist()]),
                    "y": np.array([comm_xy[c][1] for c in cids.tolist()]),
                    "r": np.array([comm_xy[c][2] for c in cids.tolist()])})
        if tiles:
            if emit:
                emit("tiles")
            src["x"], src["y"], src["pr"] = x, y, pr
            self._write_tiles(src, ra, rb, e_c, e_k.astype(np.uint8), comm_of, state,
                              file_row=file_row)
        return n_comm

    def _communities_from_arrays(self, src: dict, pr, ra, rb, e_k, e_c, sid, order):
        """Louvain over CALLS / INSTANTIATES / INHERITS (weight = confidence),
        CONTAINS (0.5) and File->File IMPORTS (0.3). Returns (community id
        per node row, (Community rows, MEMBER_OF rows)) -- written by
        _global_apply."""
        import numpy as np
        from docgraph import tiles as _tiles
        from docgraph.communities import detect_arrays
        n = len(src["ids"])
        kid = _tiles.EDGE_KIND_ID
        w = np.zeros(len(e_k), dtype=np.float64)
        for rel in ("CALLS", "INSTANTIATES", "INHERITS"):
            m = e_k == kid[rel]
            w[m] = e_c[m]
        w[e_k == kid["CONTAINS"]] = 0.5
        imp = e_k == kid["IMPORTS"]
        both_files = (ra >= 0) & (rb >= 0)
        imp &= both_files
        if imp.any():
            imp[imp] = (src["kinds"][ra[imp]] == 0) & (src["kinds"][rb[imp]] == 0)
        w[imp] = 0.3
        use = (w > 0) & both_files
        # Variables are not community members (MEMBER_OF has no Variable pair).
        if use.any():
            use[use] = (src["kinds"][ra[use]] != 3) & (src["kinds"][rb[use]] != 3)
        labels_by_kind = ["File", "Class", "Function", "Variable"]
        lab_sorted = [labels_by_kind[int(src["kinds"][i])] for i in order.tolist()]
        names_sorted = [src["names"][i] for i in order.tolist()]
        files_sorted = [src["files"][i] for i in order.tolist()]
        comms = detect_arrays(sid, lab_sorted, names_sorted, files_sorted, pr[order],
                              src["ids"][ra[use]], src["ids"][rb[use]], w[use],
                              progress=lambda s: _console.print(f"[dim]{s}[/]"))
        rows: list[dict] = []
        members: dict[str, list[dict]] = defaultdict(list)
        comm_of = np.full(n, -1, dtype=np.int64)
        for c in comms:
            cid = self._new_id()
            rows.append({
                "id": cid, "name": c.name, "size": c.size, "cohesion": float(c.cohesion),
                "top_members": json.dumps(c.top_members), "files": json.dumps(c.files),
                "pagerank": float(c.pagerank),
            })
            mem = np.asarray(c.members, dtype=np.int64)
            p = order[np.searchsorted(sid, mem)]
            comm_of[p] = cid
            for i, r in zip(mem.tolist(), p.tolist()):
                members[labels_by_kind[int(src["kinds"][r])]].append({"from_id": i, "to_id": cid})
        return comm_of, (rows, members)

    def _write_tiles(self, src: dict, ra, rb, conf, kind, comm_of, state: dict,
                     file_row=None) -> None:
        import numpy as np
        from docgraph import tiles as _tiles
        n = len(src["ids"])
        if file_row is None:
            path_row = {src["files"][i]: i for i in np.nonzero(src["kinds"] == 0)[0].tolist()}
            file_row = np.array([path_row.get(f, -1) for f in src["files"]], dtype=np.int64)
        x, y = np.asarray(src["x"], dtype=np.float64), np.asarray(src["y"], dtype=np.float64)
        miss = ~(np.isfinite(x) & np.isfinite(y))
        if miss.any():
            # nodes without a stored position (layout off at some point):
            # around their file, else on a far spiral -- never dropped
            from docgraph.layout import spiral_offsets
            fr = file_row[miss]
            base_x = np.where((fr >= 0) & np.isfinite(x[np.clip(fr, 0, None)]), x[np.clip(fr, 0, None)], np.nan)
            base_y = np.where((fr >= 0) & np.isfinite(y[np.clip(fr, 0, None)]), y[np.clip(fr, 0, None)], np.nan)
            off = spiral_offsets(int(miss.sum()))
            ext = float(np.nanmax(np.abs(np.concatenate([x[~miss], y[~miss]])))) if (~miss).any() else 0.0
            x[miss] = np.where(np.isfinite(base_x), base_x + off[:, 0] * 0.3, off[:, 0] + ext * 1.2)
            y[miss] = np.where(np.isfinite(base_y), base_y + off[:, 1] * 0.3, off[:, 1])
        if comm_of is None:
            comm_of = np.full(n, -1, dtype=np.int64)
        ok = (ra >= 0) & (rb >= 0)
        clusters = {}
        for r in self.db.community_rows():
            clusters[int(r["id"])] = {"name": r.get("name") or "", "x": r.get("x"), "y": r.get("y"),
                                      "r": r.get("r")}
        flags = np.where(src["test"], _tiles.FLAG_TEST, 0).astype(np.uint8)
        ts = _tiles.TileSource(
            ids=src["ids"], kinds=src["kinds"].astype(np.uint8), x=x, y=y,
            pagerank=np.asarray(src["pr"], dtype=np.float32), file_row=file_row,
            community=comm_of, names=[
                (nm if k != 0 else f) for nm, f, k in zip(src["names"], src["files"], src["kinds"].tolist())],
            flags=flags, edge_a=src["ids"][ra[ok]], edge_b=src["ids"][rb[ok]],
            edge_kind=np.asarray(kind)[ok], edge_conf=np.asarray(conf, dtype=np.float32)[ok],
            clusters=clusters)
        gen = int(state.get("tiles_generation", 0) or 0) + 1
        arrays, man = _tiles.build_arrays(ts, gen, progress=lambda s: _console.print(f"[dim]{s}[/]"))
        self.live.tiles = (man, arrays)
        self._publish_tiles(man, arrays)
        state["tiles_generation"] = gen
        state["tiles"] = {"generation": gen, "counts": man.get("counts"),
                          "seconds": man.get("build_seconds")}

    # ---- Persistent state (separate from cache: smaller, global) ----
    def _state_path(self) -> Path:
        return self.cfg.cache_path.parent / "state.json"

    def _load_state(self) -> dict:
        p = self._state_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _save_state(self, state: dict) -> None:
        try:
            _atomic_write_text(self._state_path(), json.dumps(state))
        except Exception:
            pass

    # ---- Tier 4 driver ----
    def _recompute_tier4(
        self,
        changed_files: set[str],
        deleted_files: set[str],
        full: bool,
        state: dict,
    ) -> None:
        """Recompute Tier 4 edges + PageRank.

        full=True: classic wipe + global recompute (used on `--full` or first
        run with empty cache). full=False: only touch edges incident to
        entities in changed_files / deleted_files. Saves the bulk of the
        write traffic on small incrementals.
        """

        if full:
            # Existing wipe-and-rebuild path
            for rel in ("SIMILAR_TO", "CO_CHANGED_WITH", "TESTS"):
                try:
                    self.db.delete_all_edges(rel)
                except Exception:
                    pass
            for label, desc in (("Function", "functions"), ("Class", "classes")):
                ids, mat = self.db.embedding_matrix(label)
                if len(ids) < 2:
                    continue
                with _bar() as prog:
                    task = prog.add_task(
                        f"SIMILAR_TO ({desc})", total=len(ids)
                    )
                    self._write_similar_edges(
                        ids, mat, label,
                        on_progress=lambda n: prog.advance(task, n),
                    )
                del ids, mat
        else:
            # Partial: the entities of changed files are new nodes (their old
            # edges went with the DETACH DELETE): top-k for them only, by id.
            # Other entities may keep stale back-links to changed entities --
            # accepted drift on incremental; a full reindex resets.
            new_ids = self.live.symtab.ids_of_files(changed_files)
            for label in ("Function", "Class"):
                ids = new_ids.get(label, [])
                if ids:
                    self._similar_for_ids(label, ids, state)

        # CO_CHANGED_WITH: skip if no git HEAD has moved since last run.
        last_heads = state.get("git_heads", {}) or {}
        cur_heads: dict[str, str] = {}
        any_head_changed = False
        for root, _prefix in self.cfg.roots_with_prefix():
            head = self._git_head(root)
            if head is None:
                # Not a git repo (or git missing) — leave as-is, treat as static
                continue
            cur_heads[str(root)] = head
            if last_heads.get(str(root)) != head:
                any_head_changed = True
        if any_head_changed or full:
            file_index = dict(self.live.symtab.file_index) or {
                r["path"]: r["id"]
                for r in self.db.fetch_all("MATCH (f:File) RETURN f.id AS id, f.path AS path")
            }
            try:
                self.db.delete_all_edges("CO_CHANGED_WITH")
            except Exception:
                pass
            total_commits = sum(
                self._count_commits(root)
                for root, _prefix in self.cfg.roots_with_prefix()
            )
            with _bar() as prog:
                task = prog.add_task(
                    "CO_CHANGED_WITH (git history)", total=max(total_commits, 1)
                )
                self._write_co_changed(
                    file_index,
                    on_progress=lambda n: prog.advance(task, n),
                )
                # Top up if rev-list count and parsed-commit count diverge so
                # the bar still finishes cleanly.
                if total_commits == 0:
                    prog.advance(task, 1)
            state["git_heads"] = cur_heads
        else:
            _console.print("[dim]CO_CHANGED_WITH: git HEAD unchanged — skipped[/]")

        # TESTS: full or partial-by-changed-files
        if full:
            function_rows_db = self.db.fetch_all(
                "MATCH (n:Function) WHERE n.is_test RETURN n.id AS id, n.name AS name, n.is_test AS is_test"
            )
            if function_rows_db:
                with _bar() as prog:
                    task = prog.add_task(
                        "TESTS edges", total=len(function_rows_db)
                    )
                    self._write_tests_edges(
                        function_rows_db, self._name_index_for(function_rows_db),
                        on_progress=lambda n: prog.advance(task, n),
                    )
        else:
            self._tests_incremental(changed_files)

        # PageRank / communities / layout / tiles: _graph_analytics().

    def _similar_for_ids(self, label: str, ids: list[int], state: dict, inline: bool = False) -> None:
        """SIMILAR_TO top-k out of the given (new) entities: one HNSW query
        each, fanned out over side connections; an exact blocked scan when
        the vector index is missing. Big batches are deferred to the host's
        maintenance when it runs one."""
        if not ids:
            return
        if not inline and self.defer_global:
            # the host computes these in background maintenance (the index
            # pass returns first); nearest-neighbour edges are advisory
            pend = state.setdefault("similar_pending", {})
            pend[label] = sorted(set(pend.get(label, [])) | {int(i) for i in ids})
            return
        sim_edges = self._similar_rows(label, ids)
        if sim_edges:
            self.db.insert_edges("SIMILAR_TO", label, label, sim_edges, validate=False)

    def _similar_rows(self, label: str, ids: list[int], db=None) -> list[dict]:
        """SIMILAR_TO rows out of `ids` (reads only -- the host's maintenance
        computes them on the read-only handle)."""
        import numpy as np
        db = db if db is not None else self.db
        fv = (self._fresh_vecs or {}).get(label) or {}
        if ids and all(int(i) in fv for i in ids):
            got = np.asarray([int(i) for i in ids], dtype=np.int64)
            mat = np.asarray([np.asarray(fv[int(i)], dtype=np.float32) for i in ids], dtype=np.float32)
        else:
            got, mat = db.embeddings_for(label, [int(i) for i in ids])
        if not len(got):
            return []
        k = int(self.cfg.similar_top_k)
        sim_edges: list[dict] = []
        if db.VECTOR_INDEXES[label] in db.list_indexes():

            def work(part):
                out = []
                with db.thread_conn():
                    for i, v in part:
                        try:
                            hits = db.vector_topk(label, v, k + 1)
                        except Exception:
                            hits = []
                        for j, score in hits:
                            if j != i and score >= 0.5:
                                out.append({"from_id": int(i), "to_id": int(j), "score": float(score)})
                return out
            items = list(zip(got.tolist(), mat))
            if len(items) <= 16:
                sim_edges = work(items)
            else:
                n_w = 4
                parts = [items[w::n_w] for w in range(n_w)]
                with ThreadPoolExecutor(max_workers=n_w) as ex:
                    for chunk in ex.map(work, parts):
                        sim_edges.extend(chunk)
        else:
            from docgraph.similar import unit_rows
            all_ids, all_mat = db.embedding_matrix(label)
            if len(all_ids) < 2:
                return []
            x = unit_rows(all_mat)
            q = unit_rows(mat)
            for s0 in range(0, len(q), 512):
                sims = q[s0:s0 + 512] @ x.T
                for r_i in range(sims.shape[0]):
                    row = sims[r_i]
                    me = int(got[s0 + r_i])
                    row[all_ids == me] = -2.0
                    top = np.argpartition(-row, min(k, len(row) - 1))[:k]
                    for j in top.tolist():
                        if row[j] >= 0.5:
                            sim_edges.append({"from_id": me, "to_id": int(all_ids[j]),
                                              "score": float(row[j])})
        return sim_edges

    def _tests_incremental(self, changed_files: set[str]) -> None:
        """TESTS for the new test functions of changed files, plus the
        unchanged tests whose target name is defined in a changed file (their
        old edges went with the recreated targets). Name lookups come from
        the live symbol table and the live test index -- no table scans."""
        live = self.live
        symtab = live.symtab
        tests = live.ensure_tests(self.db)
        new_ids = symtab.ids_of_files(changed_files)
        trows = self.db.test_functions(new_ids.get("Function", []))
        new_t = {int(r["id"]) for r in trows}
        for r in trows:
            live.add_test(int(r["id"]), r.get("name") or "")
        self._new_test_ids = new_t
        rows_f: list[dict] = []
        rows_c: list[dict] = []
        name_index = symtab.name_index

        def emit(t: int, lab: str, eid: int) -> None:
            if eid == t:
                return
            if lab == "Function":
                rows_f.append({"from_id": t, "to_id": eid})
            elif lab == "Class":
                rows_c.append({"from_id": t, "to_id": eid})
        for r in trows:
            k = test_target(r.get("name") or "")
            if k:
                for lab, eid, _f in name_index.get(k, ()):
                    emit(int(r["id"]), lab, int(eid))
        names = {name for f in changed_files for lab, _i, name, _q in symtab.by_file.get(f, ())
                 if lab in ("Function", "Class")}
        for name in names:
            for t in tests.get(name, ()):
                if t in new_t:
                    continue
                for lab, eid, f in name_index.get(name, ()):
                    if f in changed_files:
                        emit(int(t), lab, int(eid))
        if rows_f:
            self.db.insert_edges("TESTS", "Function", "Function", rows_f, validate=False)
        if rows_c:
            self.db.insert_edges("TESTS", "Function", "Class", rows_c, validate=False)

    def _git_head(self, root: Path) -> str | None:
        fast = _history.head_fast(root)
        if fast:
            return fast
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root, text=True, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    def _name_index_for(self, test_rows: list[dict]) -> dict[str, list[tuple[str, int]]]:
        """name -> [(label, id)] for just the names test functions point at
        (`test_foo` -> `foo`), via targeted queries instead of the whole
        symbol table."""
        names: set[str] = set()
        for fr in test_rows:
            stripped = fr.get("name") or ""
            for prefix in ("test_", "test"):
                if stripped.lower().startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip("_")
                    break
            if stripped:
                names.add(stripped)
        idx: dict[str, list[tuple[str, int]]] = defaultdict(list)
        lst = sorted(names)
        for s in range(0, len(lst), 5000):
            part = lst[s:s + 5000]
            for label in ("Function", "Class"):
                for r in self.db.fetch_all(
                        f"MATCH (n:{label}) WHERE n.name IN $n RETURN n.id AS id, n.name AS name",
                        {"n": part}):
                    idx[r["name"]].append((label, r["id"]))
        return idx

    # ---- Tier 4 helpers ----
    def _write_similar_edges(
        self,
        ids,
        mat,
        label: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        """Full pass: bounded-memory top-k (docgraph.similar -- exact blocks
        for small tables, IVF lists for big ones; never an n x n matrix)."""
        from docgraph.similar import top_similar
        if len(ids) < 2:
            return
        src, dst, score = top_similar(mat, int(self.cfg.similar_top_k), 0.5, on_progress=on_progress)
        if not len(src):
            return
        rows = [{"from_id": int(ids[a]), "to_id": int(ids[b]), "score": float(v)}
                for a, b, v in zip(src.tolist(), dst.tolist(), score.tolist())]
        self.db.insert_edges("SIMILAR_TO", label, label, rows)

    def _write_co_changed(
        self,
        file_index: dict[str, int],
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        for root, prefix in self.cfg.roots_with_prefix():
            try:
                out = subprocess.check_output(
                    ["git", "log", f"-{self.cfg.co_change_window}", "--name-only", "--pretty=format:---"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    creationflags=NO_WINDOW,
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
                if on_progress is not None:
                    on_progress(1)

        rows = [
            {"from_id": file_index[a], "to_id": file_index[b], "count": c}
            for (a, b), c in pair_count.items() if c >= 2
        ]
        if rows:
            self.db.insert_edges("CO_CHANGED_WITH", "File", "File", rows)

    def _count_commits(self, root: Path) -> int:
        """Cheap: just count how many `--- ` separators are in the windowed log."""
        try:
            out = subprocess.check_output(
                ["git", "rev-list", "--count", f"-{self.cfg.co_change_window}", "HEAD"],
                cwd=root, text=True, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            ).strip()
            return int(out) if out.isdigit() else 0
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return 0

    def _write_tests_edges(
        self,
        function_rows: list[dict],
        name_index: dict[str, list[tuple[str, int]]],
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        rows_func: list[dict] = []
        rows_class: list[dict] = []
        for fr in function_rows:
            if on_progress is not None:
                on_progress(1)
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

