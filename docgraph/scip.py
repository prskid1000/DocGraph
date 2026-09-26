"""Optional precise references from SCIP indexes.

If `scip-python` / `scip-typescript` are on PATH (or passed with
`--scip-python` / `--scip-typescript`), or a prebuilt index is given with
`--scip-index`, the indexer ingests the SCIP occurrences as confidence-1.0
CALLS edges (method `scip`): a reference occurrence inside function A to a
symbol whose definition is function B. Existing heuristic edges between the
same pair are upgraded in place; new pairs are inserted.

Nothing is required: with no binary and no index file this is a single
"skipped" status line.

The SCIP index is protobuf. We do not depend on the protobuf package --
`decode_index` is a ~80-line wire-format reader for the four messages we
need (Index, Document, Occurrence, SymbolInformation). Field numbers from
sourcegraph/scip `scip.proto`:

    Index             1 metadata, 2 documents, 3 external_symbols
    Document          1 relative_path, 2 occurrences, 3 symbols, 4 language
    Occurrence        1 range (packed int32), 2 symbol, 3 symbol_roles
    SymbolInformation 1 symbol, 6 display_name
"""
from __future__ import annotations

import bisect
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from docgraph.proc_util import NO_WINDOW

log = logging.getLogger(__name__)

ROLE_DEFINITION = 0x1


# ---- protobuf wire format -------------------------------------------------

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    out = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for one message. Length-
    delimited values are returned as bytes, varints as int."""
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
            yield fno, wt, v
        elif wt == 1:
            yield fno, wt, buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            yield fno, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fno, wt, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _packed_ints(v) -> list[int]:
    if isinstance(v, int):
        return [v]
    out = []
    i = 0
    while i < len(v):
        x, i = _varint(v, i)
        out.append(x)
    return out


@dataclass
class Occurrence:
    line: int              # 0-based
    symbol: str
    roles: int


@dataclass
class Document:
    path: str
    language: str = ""
    occurrences: list[Occurrence] = field(default_factory=list)


def decode_index(data: bytes) -> list[Document]:
    docs: list[Document] = []
    for fno, wt, v in _fields(data):
        if fno != 2 or wt != 2:
            continue
        doc = Document(path="")
        for dfno, dwt, dv in _fields(v):
            if dfno == 1 and dwt == 2:
                doc.path = dv.decode("utf-8", "replace").replace("\\", "/")
            elif dfno == 4 and dwt == 2:
                doc.language = dv.decode("utf-8", "replace")
            elif dfno == 2 and dwt == 2:
                rng: list[int] = []
                sym = ""
                roles = 0
                for ofno, owt, ov in _fields(dv):
                    if ofno == 1:
                        rng.extend(_packed_ints(ov))
                    elif ofno == 2 and owt == 2:
                        sym = ov.decode("utf-8", "replace")
                    elif ofno == 3 and owt == 0:
                        roles = ov
                if rng and sym:
                    doc.occurrences.append(Occurrence(line=rng[0], symbol=sym, roles=roles))
        docs.append(doc)
    return docs


# ---- mapping onto the graph ------------------------------------------------

def edges_from_documents(docs: list[Document], prefix: str,
                         spans: list[dict]) -> list[dict]:
    """Turn SCIP occurrences into (caller_id, callee_id, line) rows.

    spans: [{label, id, file, s, e}] of Function (and Class) nodes.
    A reference (non-definition occurrence of a non-local symbol) at
    `file:line` inside Function A, whose symbol is *defined* on the first
    line of Function B, yields A -> B."""
    by_file: dict[str, list[tuple[int, int, int]]] = {}
    def_line_to_fn: dict[tuple[str, int], int] = {}
    for sp in spans:
        if sp.get("label") != "Function":
            continue
        s, e = int(sp["s"] or 0), int(sp["e"] or 0)
        by_file.setdefault(sp["file"], []).append((s, e, sp["id"]))
        def_line_to_fn[(sp["file"], s)] = sp["id"]
    for lst in by_file.values():
        lst.sort()

    def enclosing(file: str, line1: int) -> int | None:
        lst = by_file.get(file)
        if not lst:
            return None
        best = None
        idx = bisect.bisect_right(lst, (line1, 1 << 60, 1 << 60))
        for s, e, fid in lst[:idx]:
            if s <= line1 <= e and (best is None or (e - s) < best[0]):
                best = (e - s, fid)
        return best[1] if best else None

    definitions: dict[str, int] = {}
    for d in docs:
        path = prefix + d.path
        for o in d.occurrences:
            if o.roles & ROLE_DEFINITION and not o.symbol.startswith("local "):
                fid = def_line_to_fn.get((path, o.line + 1))
                if fid is None:
                    # decorators / multi-line signatures: nearest def within 2 lines
                    for dl in (o.line, o.line + 2):
                        fid = def_line_to_fn.get((path, dl))
                        if fid is not None:
                            break
                if fid is not None:
                    definitions[o.symbol] = fid
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for d in docs:
        path = prefix + d.path
        for o in d.occurrences:
            if o.roles & ROLE_DEFINITION or o.symbol.startswith("local "):
                continue
            callee = definitions.get(o.symbol)
            if callee is None:
                continue
            caller = enclosing(path, o.line + 1)
            if caller is None or caller == callee:
                continue
            key = (caller, callee)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"from_id": caller, "to_id": callee, "line": o.line + 1})
    return rows


# ---- driver ----------------------------------------------------------------

_WHICH: dict[str, tuple[float, str]] = {}


def _which(name: str) -> str:
    """shutil.which, remembered for a minute (a PATH scan per index pass
    costs ~10 ms on Windows)."""
    import time as _t
    hit = _WHICH.get(name)
    now = _t.monotonic()
    if hit is not None and now - hit[0] < 60.0:
        return hit[1]
    path = shutil.which(name) or ""
    _WHICH[name] = (now, path)
    return path


def _binaries(cfg) -> list[tuple[str, str]]:
    out = []
    for name, attr in (("scip-python", "scip_python"), ("scip-typescript", "scip_typescript")):
        explicit = (getattr(cfg, attr, "") or "").strip()
        path = explicit or _which(name)
        if path:
            out.append((name, path))
    return out


def _run_binary(name: str, exe: str, root: Path, out_file: Path) -> bool:
    if name == "scip-python":
        args = [exe, "index", ".", "--project-name", root.name, "--output", str(out_file)]
    else:
        args = [exe, "index", "--output", str(out_file)]
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, text=True,
                              timeout=900, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("%s failed to run: %s", name, exc)
        return False
    if proc.returncode != 0:
        log.warning("%s exited %s: %s", name, proc.returncode, (proc.stderr or "")[-500:])
        return False
    return out_file.exists()


def maybe_ingest(cfg, db, full: bool, changed: bool, console=None) -> dict:
    mode = (getattr(cfg, "scip", "auto") or "auto").lower()
    if mode == "off":
        return {"status": "off"}
    index_files: list[tuple[Path, str]] = []     # (file, logical prefix)
    explicit = (getattr(cfg, "scip_index", "") or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = cfg.repo_root / p
        if p.exists():
            index_files.append((p, ""))
    bins = _binaries(cfg)
    if not index_files and bins and (full or changed or mode == "on"):
        for root, prefix in cfg.roots_with_prefix():
            if root == cfg.external_dir:
                continue
            for name, exe in bins:
                out = cfg.data_dir / f"{name}-{root.name}.scip"
                if _run_binary(name, exe, root, out):
                    index_files.append((out, prefix))
    if not index_files:
        # reuse indexes produced by an earlier run
        for root, prefix in cfg.roots_with_prefix():
            for name, _exe in bins:
                out = cfg.data_dir / f"{name}-{root.name}.scip"
                if out.exists():
                    index_files.append((out, prefix))
    if not index_files:
        msg = "SCIP: no scip-python / scip-typescript on PATH and no --scip-index -- skipped"
        if console is not None:
            console.print(f"[dim]{msg}[/]")
        return {"status": "skipped", "detail": msg}

    all_rows: list[dict] = []
    n_docs = 0
    files_needed: set[str] = set()
    decoded: list[tuple[list[Document], str]] = []
    for path, prefix in index_files:
        try:
            docs = decode_index(path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            log.warning("SCIP decode failed for %s: %s", path, exc)
            continue
        n_docs += len(docs)
        decoded.append((docs, prefix))
        files_needed.update(prefix + d.path for d in docs)
    spans = db.entity_spans(sorted(files_needed))
    for docs, prefix in decoded:
        all_rows.extend(edges_from_documents(docs, prefix, spans))

    existing: set[tuple[int, int]] = set()
    if all_rows:
        for r in db.fetch_all("MATCH (a:Function)-[r:CALLS]->(b:Function) RETURN a.id AS a, b.id AS b"):
            existing.add((r["a"], r["b"]))
    upgrade = [{"a": r["from_id"], "b": r["to_id"], "confidence": 1.0, "method": "scip"}
               for r in all_rows if (r["from_id"], r["to_id"]) in existing]
    new = [dict(r, confidence=1.0, method="scip")
           for r in all_rows if (r["from_id"], r["to_id"]) not in existing]
    db.upgrade_calls(upgrade)
    if new:
        db.insert_edges("CALLS", "Function", "Function", new)
    status = {"status": "ok", "indexes": [str(p) for p, _ in index_files], "documents": n_docs,
              "edges": len(all_rows), "upgraded": len(upgrade), "inserted": len(new)}
    if console is not None:
        console.print(f"[cyan]SCIP[/]: {len(all_rows)} precise edges "
                      f"({len(upgrade)} upgraded, {len(new)} new) from {n_docs} documents")
    return status


# ---- tiny encoder (tests + fixtures) ---------------------------------------

def _enc_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _enc_field(fno: int, wt: int, payload) -> bytes:
    key = _enc_varint((fno << 3) | wt)
    if wt == 0:
        return key + _enc_varint(payload)
    return key + _enc_varint(len(payload)) + payload


def encode_index(docs: list[Document]) -> bytes:
    """Encode Documents back to SCIP protobuf (used by tests)."""
    out = bytearray()
    for d in docs:
        body = bytearray(_enc_field(1, 2, d.path.encode()))
        if d.language:
            body += _enc_field(4, 2, d.language.encode())
        for o in d.occurrences:
            rng = b"".join(_enc_varint(x) for x in (o.line, 0, 1))
            occ = _enc_field(1, 2, rng) + _enc_field(2, 2, o.symbol.encode())
            if o.roles:
                occ += _enc_field(3, 0, o.roles)
            body += _enc_field(2, 2, occ)
        out += _enc_field(2, 2, bytes(body))
    return bytes(out)
