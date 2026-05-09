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
from dataclasses import asdict
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
from docgraph.db import GraphDB
from docgraph.embed import Embedder, GPU_PROVIDERS, resolve_providers
from docgraph.parse import detect_language, parse_file, FileParse, Entity, RawEdge
from docgraph.rank import compute_pagerank, write_pagerank
from docgraph.summary import build_embedding_text, chunk_body

log = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None] | None


def _gpu_providers() -> list[str]:
    return list(GPU_PROVIDERS)


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
        self.embedder = embedder or Embedder(
            cfg.embedding_model,
            providers=resolve_providers(cfg.gpu),
        )
        self._next_id = 1
        self.progress_cb: ProgressCb = None

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

    def _stream_embed_insert(
        self,
        label: str,
        plan: list[tuple[dict, str]],
        prog_label: str,
        insert_batch: int = 5000,
        cancel_token: "CancelToken | None" = None,
        progress_emit: "Callable[[int], None] | None" = None,
    ) -> None:
        """Stream-embed and insert. plan: [(row_dict, embed_text), ...].

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
            with _bar() as prog:
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
                    vecs = self.embedder.embed(
                        texts,
                        batch_size=self.cfg.embed_batch_size,
                        on_progress=_on_emb,
                    )
                    # Attach numpy slices (1.5 KB each) instead of list[float]
                    # (12 KB each) — db.insert_nodes converts to list per
                    # write batch just before the UNWIND call.
                    for r, v in zip(rows, vecs):
                        r["embedding"] = v
                    write_q.put(rows)
                    # Free the embedding text strings for this slab.
                    del texts, vecs
        finally:
            write_q.put(None)
            t.join()
        if errors:
            raise errors[0]

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
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass

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
    def index_all(self, incremental: bool = True, progress_cb: ProgressCb = None,
                  cancel_token: "CancelToken | None" = None,
                  fetch_links: bool = True, force_fetch: bool = False) -> dict:
        self.progress_cb = progress_cb
        # Cooperative cancel: poll `cancel_token.raise_if_set()` at major
        # phase boundaries. Mid-phase cancellation is unsafe (inside Kuzu
        # COPY or an ONNX embed call would corrupt state); between phases
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
        cache = load_cache(self.cfg) if incremental else {}
        # If the cache was empty AND we're "incremental", treat the run as a
        # full pass for Tier 4 purposes — there's no prior state to preserve.
        cache_was_present = bool(cache)
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
        _console.print(
            f"[cyan]Scanning[/]: {len(on_disk_rel)} files — "
            f"[green]{len(changed)}[/] changed/added, "
            f"[red]{len(deleted_rels)}[/] deleted, "
            f"[dim]{len(unchanged_rels)}[/] unchanged"
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
            # Release the Windows file lock before rmtree — Kuzu's connection
            # holds the dir open and shutil.rmtree silently leaves a partial
            # state, which then crashes the next Database() constructor with
            # `invalid unordered_map<K, T> key`.
            self.db.close()
            self.db.wipe(self.cfg.db_path)
            self.db = GraphDB(self.cfg.db_path, self.embedder.dim)
            self.db.init_schema()
            self._next_id = 1
            cache = {}
            unchanged_rels = set()
            deleted_rels = []
            changed = list(files_on_disk)

        # ---- Step 1: delete affected nodes from DB ----
        _ck()
        _emit("delete", 0, len(changed) + len(deleted_rels))
        affected = [rel for _path, rel in changed]
        self._delete_files_from_db(affected + deleted_rels)
        for rel in deleted_rels:
            cache.pop(rel, None)

        # ---- Step 2: parse changed files in parallel ----
        _ck()
        _emit("parse", 0, len(changed))
        parsed: dict[str, FileParse] = {}
        errors: list[str] = []
        if changed:
            parse_emit = _throttled("parse", len(changed))
            with _bar() as prog:
                ptask = prog.add_task("Parsing files", total=len(changed))
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
                        parse_emit(1)
                        # Per-file checkpoint: parse can be the slowest
                        # phase on big repos; this lets a cancel land
                        # within a few hundred ms instead of waiting for
                        # the whole pool to drain.
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

        # ---- Step 3a: optional LLM docstring augmentation ----
        _ck()
        if self.cfg.llm_docstrings and parsed:
            _emit("llm_augment", 0, len(parsed))
            self._augment_llm_docstrings(parsed, cancel_token=cancel_token)

        # ---- Step 3: seed ID allocator ----
        _ck()
        _emit("seed_ids")
        self._seed_ids_from_db()

        # ---- Step 4: build node rows for newly-parsed files ----
        # Streaming design: we collect (row, embed_text) plans per label and
        # then embed-and-insert in batches via _stream_embed_insert(). This
        # caps peak memory at ~one batch worth of vectors instead of holding
        # all 200k+ embeddings live at once. No `[0.0] * dim` placeholders;
        # numpy slices are attached just-in-time per batch.
        file_rows: list[dict] = []
        variable_rows: list[dict] = []
        class_plan: list[tuple[dict, str]] = []     # (row, embed_text)
        function_plan: list[tuple[dict, str]] = []  # (row, embed_text)

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
                    row = {
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": rel,
                        "line_start": ent.line_start,
                        "line_end": ent.line_end,
                        "body": ent.body,
                        "kind": ent.kind,
                        "llm_doc": ent.extra.get("llm_doc") if isinstance(ent.extra, dict) else None,
                        "pagerank": 0.0,
                    }
                    text = build_embedding_text(
                        ent.name, ent.qname, ent.signature, ent.body,
                        fp.language, ent.kind,
                        llm_doc=row["llm_doc"],
                    )
                    class_plan.append((row, text))
                elif ent.kind in ("function", "method"):
                    is_test = (
                        ent.name.startswith("test_") or
                        (ent.name.startswith("test") and len(ent.name) > 4 and ent.name[4:5].isupper()) or
                        "/test" in rel or "/tests/" in rel or "_test." in rel
                    )
                    row = {
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
                        "llm_doc": ent.extra.get("llm_doc") if isinstance(ent.extra, dict) else None,
                        "pagerank": 0.0,
                    }
                    text = build_embedding_text(
                        ent.name, ent.qname, ent.signature, ent.body,
                        fp.language, ent.kind,
                        llm_doc=row["llm_doc"],
                    )
                    function_plan.append((row, text))
                else:
                    variable_rows.append({
                        "id": eid,
                        "name": ent.name,
                        "qname": ent.qname,
                        "file": rel,
                        "line": ent.line_start,
                        "scope": ent.extra.get("scope", "module") if isinstance(ent.extra, dict) else "module",
                    })

        n_classes = len(class_plan)
        n_functions = len(function_plan)
        _console.print(
            f"[cyan]Indexing[/] {len(file_rows)} files, {n_classes} classes, "
            f"{n_functions} functions, {len(variable_rows)} variables"
        )

        # ---- Step 5a: insert files + variables (no embeddings) ----
        if file_rows:
            with _bar() as prog:
                task = prog.add_task("Writing files", total=len(file_rows))
                self.db.insert_nodes("File", file_rows, on_progress=lambda n: prog.advance(task, n))
        if variable_rows:
            with _bar() as prog:
                task = prog.add_task("Writing variables", total=len(variable_rows))
                self.db.insert_nodes("Variable", variable_rows, on_progress=lambda n: prog.advance(task, n))
        del file_rows, variable_rows

        # Warm up the embedder once before any progress bar opens, so the
        # "Loading embedding model" log doesn't punch through the live display.
        ent_total = len(class_plan) + len(function_plan)
        if class_plan or function_plan:
            _ck()
            _emit("embed_entities", 0, ent_total)
            self.embedder._ensure()
        ent_emit = _throttled("embed_entities", ent_total) if ent_total else None

        # ---- Step 5b: stream embed + insert classes/functions ----
        if class_plan:
            self._stream_embed_insert("Class", class_plan, "Embedding entities (classes)",
                                       cancel_token=cancel_token,
                                       progress_emit=ent_emit)
            class_plan.clear()
        if function_plan:
            self._stream_embed_insert("Function", function_plan, "Embedding entities (functions)",
                                       cancel_token=cancel_token,
                                       progress_emit=ent_emit)
            function_plan.clear()

        # ---- Step 5c: build chunk plan, then stream embed + insert ----
        chunk_plan: list[tuple[dict, str]] = []
        chunk_contains_func: list[dict] = []
        chunk_contains_class: list[dict] = []
        for rel, fp in parsed.items():
            for ent in fp.entities:
                eid = ent.extra.get("_id") if isinstance(ent.extra, dict) else None
                if eid is None:
                    continue
                if ent.kind not in ("function", "method", "class", "interface"):
                    continue
                pieces = chunk_body(ent.body or "", language=fp.language)
                if not pieces:
                    continue
                parent_label = "Class" if ent.kind in ("class", "interface") else "Function"
                for idx, piece in enumerate(pieces):
                    cid = self._new_id()
                    body_truncated = piece[:6000]
                    row = {
                        "id": cid,
                        "parent_qname": ent.qname,
                        "parent_label": parent_label,
                        "file": rel,
                        "idx": idx,
                        "body": body_truncated,
                    }
                    chunk_plan.append((row, body_truncated))
                    if parent_label == "Function":
                        chunk_contains_func.append({"from_id": eid, "to_id": cid})
                    else:
                        chunk_contains_class.append({"from_id": eid, "to_id": cid})

        if chunk_plan:
            _ck()
            _emit("embed_chunks", 0, len(chunk_plan))
            chunk_emit = _throttled("embed_chunks", len(chunk_plan))
            self._stream_embed_insert("Chunk", chunk_plan, "Embedding chunks",
                                       cancel_token=cancel_token,
                                       progress_emit=chunk_emit)
            chunk_plan.clear()

        # ---- Step 5d: CONTAINS_CHUNK edges ----
        n_chunk_edges = len(chunk_contains_func) + len(chunk_contains_class)
        if n_chunk_edges:
            with _bar() as prog:
                task = prog.add_task("Writing chunk edges", total=n_chunk_edges)
                self.db.insert_edges(
                    "CONTAINS_CHUNK", "Function", "Chunk", chunk_contains_func,
                    on_progress=lambda n: prog.advance(task, n),
                )
                self.db.insert_edges(
                    "CONTAINS_CHUNK", "Class", "Chunk", chunk_contains_class,
                    on_progress=lambda n: prog.advance(task, n),
                )
        del chunk_contains_func, chunk_contains_class

        # All entity bodies have been embedded + persisted; the rest of the
        # pipeline reads from `cache` and the DB. Drop `parsed` so the entity
        # body strings can be GC'd before symbol-table building loads its own
        # working set.
        changed_set = set(parsed.keys())
        n_parsed = len(parsed)
        parsed.clear()

        _ck()
        _emit("symbol_table")
        # ---- Step 7: build full symbol table from DB ----
        # qname → (label, id); name → list[(label, id, file)]
        qname_index: dict[str, tuple[str, int]] = {}
        name_index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        file_index: dict[str, int] = {}
        # Count first so the bar has a real total. Counts are cheap aggregates;
        # the row scan below is what actually takes time on big repos.
        counts: dict[str, int] = {}
        for label in ("Function", "Class", "Variable", "File"):
            rows = self.db.fetch_all(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = int(rows[0]["c"]) if rows else 0
        symtab_total = sum(counts.values())
        with _bar() as prog:
            stask = prog.add_task("Building symbol table", total=symtab_total)
            for label in ("Function", "Class", "Variable"):
                for r in self.db.fetch_all(
                    f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, "
                    f"n.qname AS qname, n.file AS file"
                ):
                    qname_index[r["qname"]] = (label, r["id"])
                    name_index[r["name"]].append((label, r["id"], r["file"]))
                    prog.advance(stask)
            for r in self.db.fetch_all("MATCH (f:File) RETURN f.id AS id, f.path AS path"):
                file_index[r["path"]] = r["id"]
                prog.advance(stask)

        # Scope-aware resolution: per-file set of files that file imports.
        # Built from the same fuzzy IMPORTS-target match we use for the edge
        # itself, so resolve() can prefer call-targets in imported files over
        # same-named symbols in unrelated files. This is the best we can do
        # without a real LSP — in practice it kills most cross-file CALLS
        # hallucinations on overloads / generics / re-exports.
        # Dedupe by target_path: many IMPORTS edges share the same target
        # (e.g. "react", "os.path"). Resolve each unique target against
        # file_index once, then fan out — cuts work from O(E × F) to
        # O(T × F + E) where T ≪ E on real repos.
        file_imports: dict[str, set[str]] = defaultdict(set)
        import_pairs: list[tuple[str, str]] = []  # (rel, target_path)
        for rel, file_data in cache.items():
            for raw in file_data.get("edges", []):
                if raw.get("kind") != "IMPORTS":
                    continue
                target_path = (raw.get("target_name") or "").replace(".", "/")
                if not target_path:
                    continue
                import_pairs.append((rel, target_path))

        unique_targets = {tp for _, tp in import_pairs}
        file_paths_list = list(file_index.keys())
        # Keep up to 2 matches per target so fan-out can skip self-matches
        # without re-scanning (preserves the original loop's semantics).
        target_matches: dict[str, list[str]] = {}
        if unique_targets:
            with _bar() as prog:
                task = prog.add_task(
                    "Resolving import scope", total=len(unique_targets)
                )
                for tp in unique_targets:
                    matches: list[str] = []
                    for path in file_paths_list:
                        if path.startswith(tp) or tp in path:
                            matches.append(path)
                            if len(matches) >= 2:
                                break
                    target_matches[tp] = matches
                    prog.advance(task)

        for rel, tp in import_pairs:
            for m in target_matches.get(tp, ()):
                if m != rel:
                    file_imports[rel].add(m)
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

        _ck()
        _emit("edges")
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
        imports_symbol_class_rows: list[dict] = []
        imports_symbol_func_rows: list[dict] = []
        overrides_rows: list[dict] = []
        module_rows_by_name: dict[str, dict] = {}

        # Pre-load existing modules so we don't duplicate
        for r in self.db.fetch_all("MATCH (m:Module) RETURN m.id AS id, m.name AS name"):
            module_rows_by_name[r["name"]] = {"id": r["id"], "name": r["name"], "language": ""}

        # Total work for the resolution bar: CONTAINS pass over changed-file
        # entities + the full RawEdge pass over every file's cached edges.
        n_resolve_contains = sum(
            len(fd.get("entities", []))
            for rel, fd in cache.items() if rel in changed_set
        )
        n_resolve_edges = sum(len(fd.get("edges", [])) for fd in cache.values())
        n_resolve_total = n_resolve_contains + n_resolve_edges

        prog_resolve = _bar() if n_resolve_total else None
        if prog_resolve is not None:
            prog_resolve.start()
            rtask = prog_resolve.add_task("Resolving edges", total=n_resolve_total)
        else:
            rtask = None  # type: ignore[assignment]

        # CONTAINS edges from cached entities (only for changed files; unchanged are still in DB)
        for rel, file_data in cache.items():
            if rel not in changed_set:
                continue
            fid = file_index.get(rel)
            if fid is None:
                if prog_resolve is not None:
                    prog_resolve.advance(rtask, len(file_data.get("entities", [])))
                continue
            # qnames in this file
            for ent_dict in file_data["entities"]:
                if prog_resolve is not None:
                    prog_resolve.advance(rtask)
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
                if prog_resolve is not None:
                    prog_resolve.advance(rtask)
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

                if kind == "IMPORTS_SYMBOL":
                    # Resolve the named symbol against the importing file's
                    # imported-file scope first, then fall back to global
                    # name match. Skip if the symbol name is too generic to
                    # disambiguate (heuristic: same name appears in 5+ files).
                    src_fid = file_index.get(src_file)
                    if src_fid is None:
                        continue
                    if not target_name:
                        continue
                    target = resolve(target_name, src_file)
                    if not target:
                        continue
                    tlabel, tid, tfile = target
                    if tlabel == "Class":
                        if needs_insert(src_file, tfile):
                            imports_symbol_class_rows.append({"from_id": src_fid, "to_id": tid})
                    elif tlabel == "Function":
                        if needs_insert(src_file, tfile):
                            imports_symbol_func_rows.append({"from_id": src_fid, "to_id": tid})
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

        # Build OVERRIDES from INHERITS edges + same-name methods. We walk the
        # inheritance closure (so a grandchild override of a grandparent method
        # is still recorded) and emit (child_method, parent_method) pairs where
        # both classes have a method of the same name. Cheap: O(classes *
        # methods) and runs once per index pass, no embeddings involved.
        # Methods are derived from qname_index by spotting Function entities
        # whose parent qname resolves to a Class — we can't read CONTAINS
        # edges from the DB yet because those rows are still in `contains_groups`
        # waiting to be written.
        if inherits_rows:
            class_methods: dict[int, dict[str, int]] = defaultdict(dict)
            for qname, (qlabel, qeid) in qname_index.items():
                if qlabel != "Function":
                    continue
                parts = qname.split("::")
                if len(parts) < 3:
                    continue
                parent_q = "::".join(parts[:-1])
                parent = qname_index.get(parent_q)
                if not parent or parent[0] != "Class":
                    continue
                class_methods[parent[1]][parts[-1]] = qeid
            # in-memory inheritance map: class_id → set(parent_class_ids)
            parents: dict[int, set[int]] = defaultdict(set)
            for ir in inherits_rows:
                parents[ir["from_id"]].add(ir["to_id"])
            # ancestors = transitive parents (avoids missing grand-overrides)
            def ancestors(cid: int, seen: set[int]) -> set[int]:
                out: set[int] = set()
                stack = list(parents.get(cid, ()))
                while stack:
                    p = stack.pop()
                    if p in seen:
                        continue
                    seen.add(p)
                    out.add(p)
                    stack.extend(parents.get(p, ()))
                return out
            seen_pairs: set[tuple[int, int]] = set()
            for child_cid, methods in class_methods.items():
                for parent_cid in ancestors(child_cid, set()):
                    pmethods = class_methods.get(parent_cid, {})
                    for mname, child_mid in methods.items():
                        parent_mid = pmethods.get(mname)
                        if parent_mid is None or parent_mid == child_mid:
                            continue
                        key = (child_mid, parent_mid)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        overrides_rows.append({"from_id": child_mid, "to_id": parent_mid})

        # Insert collected edges
        if prog_resolve is not None:
            prog_resolve.stop()
        n_edges = (
            sum(len(r) for r in contains_groups.values())
            + len(calls_rows) + len(inst_rows) + len(inherits_rows)
            + len(decorated_func_rows) + len(decorated_class_rows)
            + len(imports_file_rows) + len(imports_module_rows)
            + len(imports_symbol_class_rows) + len(imports_symbol_func_rows)
            + len(overrides_rows)
        )
        if n_edges:
            with _bar() as prog:
                task = prog.add_task("Writing graph edges", total=n_edges)
                cb = lambda n: prog.advance(task, n)
                for (fl, tl), rows in contains_groups.items():
                    self.db.insert_edges("CONTAINS", fl, tl, rows, on_progress=cb)
                self.db.insert_edges("CALLS", "Function", "Function", calls_rows, on_progress=cb)
                self.db.insert_edges("INSTANTIATES", "Function", "Class", inst_rows, on_progress=cb)
                self.db.insert_edges("INHERITS", "Class", "Class", inherits_rows, on_progress=cb)
                self.db.insert_edges("DECORATED_BY", "Function", "Function", decorated_func_rows, on_progress=cb)
                self.db.insert_edges("DECORATED_BY", "Class", "Function", decorated_class_rows, on_progress=cb)
                self.db.insert_edges("IMPORTS", "File", "File", imports_file_rows, on_progress=cb)
                self.db.insert_edges("IMPORTS", "File", "Module", imports_module_rows, on_progress=cb)
                self.db.insert_edges("IMPORTS_SYMBOL", "File", "Class", imports_symbol_class_rows, on_progress=cb)
                self.db.insert_edges("IMPORTS_SYMBOL", "File", "Function", imports_symbol_func_rows, on_progress=cb)
                self.db.insert_edges("OVERRIDES", "Function", "Function", overrides_rows, on_progress=cb)

        # ---- Step 8b: LINKS_TO edges from BFS web crawl ----
        # page_links.json is written by fetch_all whenever pages are crawled.
        # External files have path "external/<filename>" in file_index because
        # external_dir (name="external") is appended to cfg.extra_roots before
        # walk_files runs, giving it the prefix "external/".
        _page_links_file = self.cfg.external_dir / "page_links.json"
        if _page_links_file.exists() and file_index:
            try:
                _link_data = json.loads(_page_links_file.read_text(encoding="utf-8"))
                _ext_prefix = self.cfg.external_dir.name + "/"
                links_to_rows: list[dict] = []
                for _e in _link_data:
                    _fid = file_index.get(_ext_prefix + _e.get("from", ""))
                    _tid = file_index.get(_ext_prefix + _e.get("to", ""))
                    if _fid is not None and _tid is not None and _fid != _tid:
                        links_to_rows.append({"from_id": _fid, "to_id": _tid})
                if links_to_rows:
                    try:
                        self.db.execute("MATCH ()-[r:LINKS_TO]->() DELETE r")
                    except Exception:
                        pass
                    self.db.insert_edges("LINKS_TO", "File", "File", links_to_rows)
                    log.info("LINKS_TO: inserted %d hyperlink edges", len(links_to_rows))
            except Exception as _exc:
                log.warning("LINKS_TO: failed to load page_links.json: %s", _exc)

        _ck()
        _emit("tier4_pagerank")
        # ---- Step 9: Tier 4 + PageRank (incremental-aware) ----
        # Skip entirely on no-op runs (no parsed, no deleted) so a `docgraph
        # index` against an unchanged repo doesn't rewrite ~M of edges +
        # pagerank props. On partial-change incrementals, recompute only for
        # entities in changed files; on full reindex, do the global pass.
        graph_dirty = bool(changed_set) or bool(deleted_rels)
        full_recompute = (not incremental) or (not cache_was_present)
        state = self._load_state()

        if not graph_dirty and not full_recompute:
            _console.print("[dim]Tier 4 + PageRank: no changes — skipped[/]")
        else:
            self._recompute_tier4(
                changed_files=changed_set,
                deleted_files=set(deleted_rels),
                full=full_recompute,
                state=state,
            )

        # Persist state (last-known git HEAD per root, etc.)
        self._save_state(state)

        # ---- Step 10: persist cache (strip embeddings/IDs from entity dicts) ----
        # Cache entities should not carry _id (transient); strip.
        n_cache_ents = sum(len(c.get("entities", [])) for c in cache.values())
        with _bar() as prog:
            task = prog.add_task("Persisting cache", total=n_cache_ents + 1)
            for rel in cache:
                for ent in cache[rel].get("entities", []):
                    if "extra" in ent and isinstance(ent["extra"], dict):
                        ent["extra"].pop("_id", None)
                    prog.advance(task)
            save_cache(self.cfg, cache)
            prog.advance(task)

        elapsed = time.perf_counter() - t0
        total_entities = sum(len(c.get("entities", [])) for c in cache.values())
        _console.print(
            f"[green]Done[/] in {elapsed:.2f}s — "
            f"{len(on_disk_rel)} files, {total_entities} entities, "
            f"{n_parsed} reparsed, {len(deleted_rels)} deleted, {len(errors)} errors"
        )
        _emit("done", n_parsed, len(on_disk_rel))
        return {
            "files": len(on_disk_rel),
            "changed": n_parsed,
            "deleted": len(deleted_rels),
            "entities": sum(len(c.get("entities", [])) for c in cache.values()),
            "elapsed": elapsed,
            "errors": len(errors),
        }

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
            self._state_path().write_text(json.dumps(state))
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
        dirty_files = changed_files | deleted_files

        if full:
            # Existing wipe-and-rebuild path
            try:
                self.db.execute("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
                self.db.execute("MATCH ()-[r:CO_CHANGED_WITH]->() DELETE r")
                self.db.execute("MATCH ()-[r:TESTS]->() DELETE r")
            except Exception:
                pass
            for label, desc in (("Function", "functions"), ("Class", "classes")):
                rows = self.db.fetch_all(
                    f"MATCH (n:{label}) RETURN n.id AS id, n.embedding AS embedding"
                )
                if len(rows) < 2:
                    continue
                with _bar() as prog:
                    task = prog.add_task(
                        f"SIMILAR_TO ({desc})", total=len(rows)
                    )
                    self._write_similar_edges(
                        rows, label,
                        on_progress=lambda n: prog.advance(task, n),
                    )
        else:
            # Partial: only entities in dirty_files have changed embeddings.
            # Delete SIMILAR_TO incident to those entities, then recompute
            # outgoing top-K only for them. Other entities may keep stale
            # back-links to changed entities — accepted drift on incremental;
            # full reindex resets.
            for label, desc in (("Function", "functions"), ("Class", "classes")):
                # Pre-count dirty entities so we can size the bar honestly
                if not dirty_files:
                    continue
                files_list = list(dirty_files)
                cnt_rows = self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.file IN $files RETURN count(n) AS c",
                    {"files": files_list},
                )
                total = int(cnt_rows[0]["c"]) if cnt_rows else 0
                if total == 0:
                    continue
                with _bar() as prog:
                    task = prog.add_task(
                        f"SIMILAR_TO ({desc}, partial)", total=total
                    )
                    self._recompute_similar_partial(
                        label, dirty_files,
                        on_progress=lambda n: prog.advance(task, n),
                    )

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
            file_index = {
                r["path"]: r["id"]
                for r in self.db.fetch_all("MATCH (f:File) RETURN f.id AS id, f.path AS path")
            }
            try:
                self.db.execute("MATCH ()-[r:CO_CHANGED_WITH]->() DELETE r")
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
                "MATCH (n:Function) RETURN n.id AS id, n.name AS name, n.is_test AS is_test"
            )
            if function_rows_db:
                with _bar() as prog:
                    task = prog.add_task(
                        "TESTS edges", total=len(function_rows_db)
                    )
                    self._write_tests_edges(
                        function_rows_db, self._build_name_index(),
                        on_progress=lambda n: prog.advance(task, n),
                    )
        else:
            if dirty_files:
                files_list = list(dirty_files)
                cnt = self.db.fetch_all(
                    "MATCH (n:Function) WHERE n.file IN $files AND n.is_test "
                    "RETURN count(n) AS c",
                    {"files": files_list},
                )
                total = int(cnt[0]["c"]) if cnt else 0
                if total:
                    with _bar() as prog:
                        task = prog.add_task("TESTS edges (partial)", total=total)
                        self._recompute_tests_partial(
                            dirty_files,
                            on_progress=lambda n: prog.advance(task, n),
                        )

        # PageRank: gated by graph_dirty at the caller, so always run here.
        # Inherently global — every node's rank depends on the whole graph
        # topology, so partial recompute would be approximate. Keep full.
        # NetworkX exposes no per-iteration hook so the bar fills in two
        # steps: 0% before compute_pagerank, 50% after, 100% after write.
        with _bar() as prog:
            task = prog.add_task("PageRank", total=2)
            scores = compute_pagerank(self.db)
            prog.advance(task)
            write_pagerank(self.db, scores)
            prog.advance(task)

    def _git_head(self, root: Path) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    def _build_name_index(self) -> dict[str, list[tuple[str, int]]]:
        idx: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for r in self.db.fetch_all("MATCH (n:Function) RETURN n.id AS id, n.name AS name"):
            idx[r["name"]].append(("Function", r["id"]))
        for r in self.db.fetch_all("MATCH (n:Class) RETURN n.id AS id, n.name AS name"):
            idx[r["name"]].append(("Class", r["id"]))
        return idx

    def _recompute_similar_partial(
        self,
        label: str,
        dirty_files: set[str],
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        if not dirty_files:
            return
        files_list = list(dirty_files)
        # Edges to delete: any SIMILAR_TO touching an entity in a dirty file
        try:
            self.db.execute(
                f"MATCH (a:{label})-[r:SIMILAR_TO]->(b:{label}) "
                f"WHERE a.file IN $files OR b.file IN $files DELETE r",
                {"files": files_list},
            )
        except Exception:
            pass
        # Get the dirty entity IDs to recompute outgoing top-K for
        dirty_id_rows = self.db.fetch_all(
            f"MATCH (n:{label}) WHERE n.file IN $files RETURN n.id AS id",
            {"files": files_list},
        )
        dirty_ids = {r["id"] for r in dirty_id_rows}
        if not dirty_ids:
            return
        rows = self.db.fetch_all(
            f"MATCH (n:{label}) RETURN n.id AS id, n.embedding AS embedding"
        )
        if len(rows) < 2:
            return
        import numpy as np
        ids = [r["id"] for r in rows]
        id_to_idx = {eid: i for i, eid in enumerate(ids)}
        try:
            mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
        except Exception:
            return
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        update_idxs = [id_to_idx[i] for i in dirty_ids if i in id_to_idx]
        if not update_idxs:
            return
        sub = mat[update_idxs]
        sims = sub @ mat.T
        sim_edges: list[dict] = []
        k = self.cfg.similar_top_k
        n = len(ids)
        top_k = min(k, n - 1)
        if top_k <= 0:
            return
        for row_i, idx in enumerate(update_idxs):
            row = sims[row_i].copy()
            row[idx] = -1.0
            top = np.argpartition(-row, top_k)[:top_k]
            for j in top:
                score = float(row[int(j)])
                if score < 0.5:
                    continue
                sim_edges.append({
                    "from_id": ids[idx],
                    "to_id": ids[int(j)],
                    "score": score,
                })
            if on_progress is not None:
                on_progress(1)
        if sim_edges:
            self.db.insert_edges("SIMILAR_TO", label, label, sim_edges)

    def _recompute_tests_partial(
        self,
        dirty_files: set[str],
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        if not dirty_files:
            return
        files_list = list(dirty_files)
        # Drop any TESTS edge incident to a changed file
        for from_to in (
            "(a:Function)-[r:TESTS]->(b:Function)",
            "(a:Function)-[r:TESTS]->(b:Class)",
        ):
            try:
                self.db.execute(
                    f"MATCH {from_to} WHERE a.file IN $files OR b.file IN $files DELETE r",
                    {"files": files_list},
                )
            except Exception:
                pass
        # Re-link tests in changed files. The tests-in-unchanged-files that
        # may now point to changed entities are out of scope on incremental.
        test_rows = self.db.fetch_all(
            "MATCH (n:Function) WHERE n.file IN $files AND n.is_test "
            "RETURN n.id AS id, n.name AS name, n.is_test AS is_test",
            {"files": files_list},
        )
        if not test_rows:
            return
        self._write_tests_edges(
            test_rows, self._build_name_index(),
            on_progress=on_progress,
        )

    # ---- Tier 4 helpers ----
    def _write_similar_edges(
        self,
        rows: list[dict],
        label: str,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
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
            if on_progress is not None:
                on_progress(end - start)
        if sim_edges:
            self.db.insert_edges("SIMILAR_TO", label, label, sim_edges)

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

