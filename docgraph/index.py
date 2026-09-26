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
from docgraph.db import GraphDB, SCHEMA_VERSION
from docgraph.embed import Embedder, resolve_device
from docgraph.parse import classify_file, parse_file, FileParse, Entity, RawEdge
from docgraph.rank import compute_pagerank, write_pagerank
from docgraph.resolve import ModuleIndex, SymbolResolver, candidate_confidence
from docgraph.summary import build_embedding_text, chunk_body
from docgraph.proc_util import NO_WINDOW
from docgraph import history as _history
from docgraph import merkle as _merkle

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
                kind = classify_file(full)
                if kind is None:
                    continue
                if kind.startswith("text:") and not text_fallback:
                    continue
                out.append((full, f"{prefix}{rel}"))
    return out


# --- Parse worker ---------------------------------------------------------


def _parse_worker(args: tuple) -> dict | None:
    file_path, repo_root, rel_override = args[:3]
    text_fallback = bool(args[3]) if len(args) > 3 else True
    try:
        fp = parse_file(Path(file_path), Path(repo_root), rel_override=rel_override,
                        text_fallback=text_fallback)
        if fp is None:
            return None
        return {
            "file": fp.file,
            "language": fp.language,
            "lines": fp.lines,
            "entities": [asdict(e) for e in fp.entities],
            "edges": [asdict(e) for e in fp.edges],
            "chunks": fp.chunks,
            "extra": fp.extra,
        }
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{file_path}: {e}"}


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


def load_cache(cfg: Config) -> dict[str, dict]:
    if not cfg.cache_path.exists():
        return {}
    try:
        import orjson
        return orjson.loads(cfg.cache_path.read_bytes())
    except Exception:
        return {}


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a temp file + atomic rename, so a process
    killed mid-write (reaper, crash, power loss) can never leave a truncated
    file. A torn cache.json would make `load_cache` throw → return {} →
    force a full reindex of everything. os.replace is atomic on Windows
    and POSIX."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_cache(cfg: Config, cache: dict[str, dict]) -> None:
    import orjson
    tmp = cfg.cache_path.with_name(cfg.cache_path.name + ".tmp")
    tmp.write_bytes(orjson.dumps(cache))
    os.replace(tmp, cfg.cache_path)


# --- Indexer --------------------------------------------------------------


class Indexer:
    def __init__(self, cfg: Config, db: GraphDB, embedder: Embedder | None = None):
        self.cfg = cfg
        self.db = db
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
                            embed_emit: Callable[[int], None], counts: dict) -> None:
        """Rows for one parsed batch -> File/Variable inserts, embedded
        Class/Function/Chunk inserts and CONTAINS_CHUNK edges."""
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
        self.db.insert_edges("CONTAINS_CHUNK", "Function", "Chunk", cc_func)
        self.db.insert_edges("CONTAINS_CHUNK", "Class", "Chunk", cc_class)
        self.db.insert_edges("CONTAINS_CHUNK", "File", "Chunk", cc_file)

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

    def _delete_files_from_db(self, files: list[str]) -> None:
        """DETACH DELETE all entities + the File node for each path."""
        if not files:
            return
        # Delete entities first (matches all by .file property)
        for label in ("Function", "Class", "Variable", "Route", "Tool"):
            if label in ("Route", "Tool") and not self.db.has_table(label):
                continue
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
        state0 = self._load_state()
        if incremental and cache and state0.get("schema_version") != SCHEMA_VERSION:
            # The DB schema or the cache entry shape changed since this root
            # was indexed (edges without confidence, nodes without ehash,
            # raw edges without receivers ...). An incremental pass would mix
            # both shapes, so rebuild once.
            _console.print(
                f"[yellow]Index format v{state0.get('schema_version', 1)} -> "
                f"v{SCHEMA_VERSION}: running a full reindex[/]"
            )
            log.warning("schema v%s -> v%s: forcing full reindex of %s",
                        state0.get("schema_version", 1), SCHEMA_VERSION, self.cfg.repo_root)
            incremental = False
            cache = {}
        # If the cache was empty AND we're "incremental", treat the run as a
        # full pass for Tier 4 purposes — there's no prior state to preserve.
        cache_was_present = bool(cache)
        files_on_disk = walk_files(self.cfg)
        # logical_rel → absolute path
        on_disk_rel: dict[str, Path] = {rel: path for path, rel in files_on_disk}

        # Compute hashes via the file-hash tree: files whose (size, mtime)
        # match the previous scan reuse their stored hash without a read.
        merkle_path = self.cfg.data_dir / "merkle.json"
        prev_tree = _merkle.load(merkle_path) if (incremental and cache) else {}
        scan_res, scan_stats = _merkle.scan(files_on_disk, prev_tree)
        changed: list[tuple[Path, str]] = []  # (absolute_path, logical_rel)
        unchanged_rels: set[str] = set()
        new_hashes: dict[str, str] = scan_res.hashes
        for rel, path in on_disk_rel.items():
            h = new_hashes.get(rel)
            if h is None:
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
            f"[dim]{len(unchanged_rels)}[/] unchanged "
            f"[dim](hashed {scan_res.hashed}, stat-reused {scan_res.reused})[/]"
        )

        # No changes: bail
        if incremental and not changed and not deleted_rels:
            try:
                _merkle.save(merkle_path, scan_res, scan_stats)
            except Exception:
                pass
            hist_updated = self._refresh_pending_history(state0, set())
            if hist_updated:
                self._save_state(state0)
            return {
                "files": len(on_disk_rel), "changed": 0, "deleted": 0,
                "entities": sum(len(c.get("entities", [])) for c in cache.values()),
                "elapsed": time.perf_counter() - t0, "errors": 0,
                "hashed": scan_res.hashed, "hash_reused": scan_res.reused,
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
        # Embedding cache: harvest the vectors of every node we are about to
        # delete, keyed by content hash, so moved / renamed / unchanged-body
        # entities are not re-embedded below.
        vec_cache: dict[str, dict[str, list]] = {}
        if incremental and getattr(self.cfg, "embed_cache", True) and (affected or deleted_rels):
            for label in ("Function", "Class", "Chunk"):
                try:
                    vec_cache[label] = self.db.embeddings_in_files(label, affected + deleted_rels)
                except Exception:
                    vec_cache[label] = {}
        # Old entities of affected files, to record removed symbols later.
        old_entities: dict[str, list[tuple[str, str, str]]] = {}
        for rel in affected + deleted_rels:
            ents = (cache.get(rel) or {}).get("entities") or []
            if ents:
                old_entities[rel] = [(e.get("qname", ""), e.get("name", ""), e.get("kind", ""))
                                     for e in ents]
        self._delete_files_from_db(affected + deleted_rels)
        for rel in deleted_rels:
            cache.pop(rel, None)

        # ---- Steps 2-5: parse -> rows -> embed -> insert, streamed in file batches ----
        # Memory stays flat on big repos: each batch of `index_batch_files`
        # files is parsed, blamed, (optionally) LLM-augmented, turned into
        # node rows, embedded and written before the next batch is parsed, so
        # at most one batch of bodies / vectors is alive at a time. Ids come
        # from one allocator seeded from the DB state after the delete step.
        _ck()
        _emit("seed_ids")
        self._seed_ids_from_db()
        errors: list[str] = []
        changed_set: set[str] = set()
        n_parsed = 0
        use_cache = incremental and getattr(self.cfg, "embed_cache", True)
        text_fallback = bool(getattr(self.cfg, "text_fallback", True))
        batch_files = max(50, int(getattr(self.cfg, "index_batch_files", 2000) or 2000))
        blame_budget = int(getattr(self.cfg, "history_max_files", 5000) or 0)
        counts = {"files": 0, "classes": 0, "functions": 0, "variables": 0, "chunks": 0}
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

            with _bar() as prog, ProcessPoolExecutor(max_workers=self.cfg.workers) as ex:
                ptask = prog.add_task("Parsing files", total=len(changed))
                etask = prog.add_task("Embedding", total=0)
                for b0 in range(0, len(args_all), batch_files):
                    _ck()
                    batch_args = args_all[b0:b0 + batch_files]
                    parsed: dict[str, FileParse] = {}
                    for result in ex.map(_parse_worker, batch_args, chunksize=8):
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
                        cache[fp.file] = {
                            "hash": new_hashes[fp.file],
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
                    # 3a: optional LLM docstring augmentation
                    _ck()
                    if self.cfg.llm_docstrings:
                        _emit("llm_augment", 0, len(parsed))
                        self._augment_llm_docstrings(parsed, cancel_token=cancel_token)
                    # 4-5: rows, embeddings, inserts for this batch
                    self._write_parsed_batch(
                        parsed, new_hashes, blame_map, vec_cache, use_cache,
                        cancel_token, prog, etask, embed_state, _embed_emit, counts,
                    )
                    changed_set.update(parsed.keys())
                    n_parsed += len(parsed)
                    parsed.clear()
            _emit("embed_entities", embed_state["done"], embed_state["total"])
        _console.print(
            f"[cyan]Indexed[/] {counts['files']} files, {counts['classes']} classes, "
            f"{counts['functions']} functions, {counts['variables']} variables, "
            f"{counts['chunks']} chunks"
        )
        del vec_cache

        # Search indexes (HNSW vectors + FTS). A fresh DB gets them once,
        # after the bulk load; an existing DB keeps them current by itself.
        _ck()
        _emit("search_index")
        search_status = self.db.ensure_search_indexes(
            on_progress=lambda what: _console.print(f"[dim]Building {what}[/]"))

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

        # Scope-aware resolution (resolve.py). ModuleIndex maps import
        # strings to files (dotted / relative / JS specifiers, suffix
        # matched); from the cached raw edges we build, per file:
        #   file_imports  files it imports (module-level)
        #   import_map    {symbol: files} for `from m import X` / `import {X}`
        #   recv_map      {alias: files} for module aliases used as receivers
        #   aliases       {alias: original} for `import X as Y`
        # The SymbolResolver cascade turns each call site into an edge with a
        # confidence + method, or keeps the candidates when it stays ambiguous.
        mod_index = ModuleIndex(list(file_index.keys()))
        file_imports: dict[str, set[str]] = defaultdict(set)
        import_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        recv_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        aliases: dict[str, dict[str, str]] = defaultdict(dict)
        external: dict[str, set[str]] = defaultdict(set)
        for rel, file_data in cache.items():
            for raw in file_data.get("edges", []):
                kind = raw.get("kind")
                if kind == "IMPORTS":
                    mod = raw.get("target_name") or ""
                    files = [f for f in mod_index.resolve(mod, rel) if f != rel]
                    if not files:
                        m = mod.strip("'\"`")
                        if m and not m.startswith("."):
                            external[rel].add(m.split(".", 1)[0].split("/", 1)[0])
                            external[rel].add(m.rsplit(".", 1)[-1].rsplit("/", 1)[-1])
                        continue
                    file_imports[rel].update(files)
                    m = mod.strip("'\"`")
                    tail = m.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in m else m.rsplit(".", 1)[-1]
                    if tail:
                        recv_map[rel][tail].update(files)
                    recv_map[rel][m].update(files)
                elif kind == "IMPORTS_SYMBOL":
                    ex = raw.get("extra") or {}
                    sym = raw.get("target_name") or ""
                    mod = ex.get("module") or ""
                    alias = ex.get("alias") or ""
                    if not sym:
                        continue
                    if alias:
                        aliases[rel][alias] = sym
                    if not mod:
                        continue
                    files = [f for f in mod_index.resolve(mod, rel) if f != rel]
                    if not files and not mod.startswith("."):
                        external[rel].add(alias or sym)
                    if files:
                        import_map[rel][sym].update(files)
                        if alias:
                            import_map[rel][alias].update(files)
                        # JS default imports name the module itself
                        if mod.startswith(".") and "/" in mod:
                            recv_map[rel][alias or sym].update(files)
                    # `from pkg import submodule` -> submodule usable as receiver
                    sub = mod_index.resolve(f"{mod}.{sym}" if not mod.startswith("./") else f"{mod}/{sym}", rel)
                    if sub:
                        recv_map[rel][alias or sym].update(sub)
                        file_imports[rel].update(f for f in sub if f != rel)

        resolver = SymbolResolver(
            name_index, file_imports,
            import_map={k: dict(v) for k, v in import_map.items()},
            recv_map={k: dict(v) for k, v in recv_map.items()},
            aliases=dict(aliases),
            method_ids={i for q, (lab, i) in qname_index.items()
                        if lab == "Function" and q.count("::") >= 2},
            external={k: v for k, v in external.items()},
        )

        def needs_insert(src_file: str, target_file: str | None) -> bool:
            """True if either endpoint was just (re)created."""
            if src_file in changed_set:
                return True
            if target_file and target_file in changed_set:
                return True
            return False

        def resolve(name: str, src_file: str, prefer_kind: str | None = None,
                    raw_extra: dict | None = None):
            ex = raw_extra or {}
            return resolver.resolve(name, src_file, prefer_kind=prefer_kind,
                                    recv=ex.get("recv"), attr=bool(ex.get("attr")))

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

        # Framework routes / MCP tools: node rows for changed files, HANDLES
        # edges for any route whose route-file or handler-file changed.
        route_rows: list[dict] = []
        tool_rows: list[dict] = []
        handles_route_rows: list[dict] = []
        handles_tool_rows: list[dict] = []
        try:
            existing_fw = self.db.framework_nodes()
        except Exception:
            existing_fw = {}
        calls_cand_rows: list[dict] = []

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
                rex = raw.get("extra") or {}

                if kind == "IMPORTS":
                    src_fid = file_index.get(src_file)
                    if src_fid is None:
                        continue
                    matched = [f for f in mod_index.resolve(target_name or "", src_file)
                               if f != src_file and f in file_index]
                    if matched:
                        for mpath in matched[:2]:
                            if needs_insert(src_file, mpath):
                                imports_file_rows.append({"from_id": src_fid, "to_id": file_index[mpath]})
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
                    # The symbol's own module (extra.module) makes this an
                    # import_map hit in the cascade; otherwise the usual
                    # same-file -> imported-file -> global order applies.
                    src_fid = file_index.get(src_file)
                    if src_fid is None:
                        continue
                    if not target_name:
                        continue
                    res = resolve(target_name, src_file)
                    if res.target is None:
                        continue
                    tlabel, tid, tfile = res.target
                    if tlabel == "Class":
                        if needs_insert(src_file, tfile):
                            imports_symbol_class_rows.append({"from_id": src_fid, "to_id": tid})
                    elif tlabel == "Function":
                        if needs_insert(src_file, tfile):
                            imports_symbol_func_rows.append({"from_id": src_fid, "to_id": tid})
                    continue

                if kind in ("ROUTE", "TOOL"):
                    # Handler: exact qname from a decorator, else a name the
                    # cascade resolves (registration style), else nothing.
                    hid = hfile = None
                    if src_qname and src_qname in qname_index and qname_index[src_qname][0] == "Function":
                        hid, hfile = qname_index[src_qname][1], src_file
                    elif target_name:
                        res = resolve(target_name, src_file, prefer_kind="Function")
                        if res.target is not None and res.target[0] == "Function":
                            hid, hfile = res.target[1], res.target[2]
                    label = "Route" if kind == "ROUTE" else "Tool"
                    name = rex.get("name") or target_name or ""
                    if src_file in changed_set:
                        nid = self._new_id()
                        if label == "Route":
                            route_rows.append({
                                "id": nid, "name": name, "method": rex.get("method") or "",
                                "path": rex.get("path") or "", "framework": rex.get("framework") or "",
                                "file": src_file, "line": int(line or 0),
                                "handler": src_qname or target_name or "",
                            })
                        else:
                            tool_rows.append({
                                "id": nid, "name": name, "kind": rex.get("tool_kind") or "tool",
                                "framework": rex.get("framework") or "mcp",
                                "file": src_file, "line": int(line or 0),
                                "handler": src_qname or target_name or "",
                            })
                    else:
                        found = existing_fw.get((src_file, int(line or 0), name))
                        nid = found[1] if found else None
                        if nid is None or not (hfile and hfile in changed_set):
                            continue
                    if hid is not None and nid is not None:
                        (handles_route_rows if label == "Route" else handles_tool_rows).append(
                            {"from_id": nid, "to_id": hid})
                    continue

                if not src_qname or src_qname not in qname_index:
                    continue
                src_label, src_id = qname_index[src_qname]

                if kind == "CALLS":
                    if src_label != "Function":
                        continue
                    res = resolve(target_name, src_file, prefer_kind="Function", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Function":
                        if needs_insert(src_file, res.target[2]):
                            calls_rows.append({"from_id": src_id, "to_id": res.target[1], "line": line,
                                               "confidence": float(res.confidence),
                                               "method": res.method})
                        alts = [c for c in res.candidates if c[0] == "Function" and c[1] != res.target[1]]
                        for c in alts[:4]:
                            if needs_insert(src_file, c[2]):
                                calls_cand_rows.append({"from_id": src_id, "to_id": c[1], "line": line,
                                                        "confidence": candidate_confidence(len(alts) + 1),
                                                        "method": "alternative"})
                    elif res.candidates:
                        cands = [c for c in res.candidates if c[0] == "Function"]
                        for c in cands[:5]:
                            if needs_insert(src_file, c[2]):
                                calls_cand_rows.append({"from_id": src_id, "to_id": c[1], "line": line,
                                                        "confidence": candidate_confidence(len(cands)),
                                                        "method": res.method})
                elif kind == "INSTANTIATES":
                    if src_label != "Function":
                        continue
                    res = resolve(target_name, src_file, prefer_kind="Class", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Class":
                        if needs_insert(src_file, res.target[2]):
                            inst_rows.append({"from_id": src_id, "to_id": res.target[1], "line": line,
                                              "confidence": float(res.confidence), "method": res.method})
                elif kind == "INHERITS":
                    if src_label != "Class":
                        continue
                    res = resolve(target_name, src_file, prefer_kind="Class", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Class":
                        if needs_insert(src_file, res.target[2]):
                            inherits_rows.append({"from_id": src_id, "to_id": res.target[1],
                                                  "confidence": float(res.confidence),
                                                  "method": res.method})
                elif kind == "DECORATED_BY":
                    res = resolve(target_name, src_file, prefer_kind="Function", raw_extra=rex)
                    if res.target is not None and res.target[0] == "Function":
                        if needs_insert(src_file, res.target[2]):
                            if src_label == "Function":
                                decorated_func_rows.append({"from_id": src_id, "to_id": res.target[1]})
                            elif src_label == "Class":
                                decorated_class_rows.append({"from_id": src_id, "to_id": res.target[1]})

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
            + len(overrides_rows) + len(calls_cand_rows)
            + len(handles_route_rows) + len(handles_tool_rows)
        )
        if route_rows:
            self.db.insert_nodes("Route", route_rows)
        if tool_rows:
            self.db.insert_nodes("Tool", tool_rows)
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
                self.db.insert_edges("CALLS_CANDIDATE", "Function", "Function", calls_cand_rows, on_progress=cb)
                self.db.insert_edges("HANDLES", "Route", "Function", handles_route_rows, on_progress=cb)
                self.db.insert_edges("HANDLES", "Tool", "Function", handles_tool_rows, on_progress=cb)
        resolution_stats = dict(resolver.stats)

        # ---- Step 8a: optional precise references (SCIP) ----
        scip_status: dict = {}
        try:
            from docgraph import scip as _scip
            _ck()
            scip_status = _scip.maybe_ingest(self.cfg, self.db, full=not incremental or not cache_was_present,
                                             changed=bool(changed_set) or bool(deleted_rels),
                                             console=_console)
        except Exception as exc:  # noqa: BLE001 - never fail the index over SCIP
            log.warning("SCIP ingest failed: %s", exc)
            scip_status = {"status": "error", "detail": str(exc)}

        # ---- Step 8c: history of unchanged files blamed while uncommitted ----
        state_hist = self._load_state()
        pending_before = set(state_hist.get("history_pending") or [])
        new_pending = {rel for rel, b in blame_map.items()
                       if any(sha == _history.UNCOMMITTED for sha, _ in b)}
        removed = list(state_hist.get("removed_symbols") or [])
        now_ts = time.time()
        head_now = self._primary_head()
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

        n_communities = None
        if not graph_dirty and not full_recompute:
            _console.print("[dim]Tier 4 + PageRank: no changes — skipped[/]")
        else:
            self._recompute_tier4(
                changed_files=changed_set,
                deleted_files=set(deleted_rels),
                full=full_recompute,
                state=state,
            )
            if getattr(self.cfg, "communities", True):
                _ck()
                _emit("communities")
                try:
                    n_communities = self._recompute_communities()
                except Exception as exc:  # noqa: BLE001
                    log.warning("community detection failed: %s", exc)

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
        state["scan"] = {"hashed": scan_res.hashed, "reused": scan_res.reused,
                         "root_hash": scan_res.root_hash}
        if n_communities is not None:
            state["communities"] = n_communities
        if scip_status:
            state["scip"] = scip_status
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
        try:
            _merkle.save(merkle_path, scan_res, scan_stats)
        except Exception:
            log.debug("merkle save failed", exc_info=True)

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
            "hashed": scan_res.hashed,
            "hash_reused": scan_res.reused,
            "embed_cache_hits": self.embed_cache_hits,
            "embedded": self.embed_cache_misses,
            "communities": n_communities if n_communities is not None else -1,
        }

    # ---- History helpers ----
    def _owner(self, rel: str) -> tuple[Path, str] | None:
        for root, prefix in self.cfg.roots_with_prefix():
            if prefix == "":
                return root, rel
            if rel.startswith(prefix):
                return root, rel[len(prefix):]
        return None

    def _git_roots(self) -> set[Path]:
        if not hasattr(self, "_git_roots_cache"):
            roots = set()
            for root, _p in self.cfg.roots_with_prefix():
                if root == self.cfg.external_dir:
                    continue
                if _history.is_git_repo(root):
                    roots.add(root)
            self._git_roots_cache = roots
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

    # ---- Communities ----
    def _recompute_communities(self) -> int:
        from docgraph.communities import detect
        nodes, edges = self.db.community_graph()
        comms = detect(nodes, edges)
        rows: list[dict] = []
        members: dict[str, list[dict]] = defaultdict(list)
        for c in comms:
            cid = self._new_id()
            rows.append({
                "id": cid, "name": c.name, "size": c.size, "cohesion": float(c.cohesion),
                "top_members": json.dumps(c.top_members), "files": json.dumps(c.files),
                "pagerank": float(c.pagerank),
            })
            for m in c.members:
                lab = nodes[m]["label"]
                members[lab].append({"from_id": m, "to_id": cid})
        self.db.replace_communities(rows, members)
        _console.print(f"[cyan]Communities[/]: {len(rows)} detected")
        return len(rows)

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
                creationflags=NO_WINDOW,
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

