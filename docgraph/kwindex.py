"""Keyword (BM25) index for search -- replaces Kuzu's FTS extension.

Why not Kuzu FTS: in Kuzu 0.11 deleting a row that was inserted after the
FTS index was built crashes the process (access violation), rows loaded
with COPY are invisible to it, and it makes every DETACH DELETE ~5x slower.
An incremental indexer deletes and re-inserts rows all the time, so the
keyword side lives here instead (derived data, rebuildable from Kuzu).

Per label (Function / Class / Chunk):

* base   immutable CSR postings sorted by a stable 64-bit term hash:
         term_hash[T], offs[T+1], docs[P] (slot), tfs[P]; doc_ids[S] (sorted),
         doc_len[S]. Persisted as `.docgraph/kw/<label>.npz`.
* delta  docs added since the base (term -> {id: tf}), persisted as
         `<label>.delta.json` -- small, rewritten per incremental pass.
* dead   tombstones: base slots / delta ids removed since the base.

Query = BM25 over base (vectorised: np.add.at over the query terms'
posting slices) + delta, minus tombstones; top-k by argpartition. compact()
folds delta + tombstones into a new base (full index, or maintenance).
Tokenizer = bm25.tokenize (camelCase / snake_case split, lowercased).
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from docgraph.bm25 import tokenize

LABELS = ("Function", "Class", "Chunk")
MAX_DOC_CHARS = 6000
K1, B = 1.2, 0.75
_HCACHE: dict[str, int] = {}


def term_hash(t: str) -> int:
    h = _HCACHE.get(t)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=8).digest(), "little")
        if len(_HCACHE) < 2_000_000:
            _HCACHE[t] = h
    return h


def doc_terms(*parts: str) -> Counter:
    c: Counter = Counter()
    for p in parts:
        if p:
            c.update(tokenize(p[:MAX_DOC_CHARS]))
    return c


class LabelIndex:
    def __init__(self, label: str, directory: Path):
        self.label = label
        self.dir = Path(directory)
        self.lock = threading.RLock()
        self._empty_base()
        self.delta: dict[int, dict[str, int]] = {}         # id -> {term: tf}
        self.delta_post: dict[str, dict[int, int]] = defaultdict(dict)
        self.dead_slots = np.zeros(0, dtype=bool)
        self.dead_ids: set[int] = set()                     # removed base ids (for persistence)
        self.dirty = False

    def _empty_base(self) -> None:
        self.term_hash = np.zeros(0, np.uint64)
        self.offs = np.zeros(1, np.int64)
        self.docs = np.zeros(0, np.int32)
        self.tfs = np.zeros(0, np.uint16)
        self.doc_ids = np.zeros(0, np.int64)
        self.doc_len = np.zeros(0, np.int32)
        self.dead_slots = np.zeros(0, dtype=bool)

    # ---- sizes ----
    @property
    def n_docs(self) -> int:
        return int(len(self.doc_ids) - int(self.dead_slots.sum()) + len(self.delta))

    def _avgdl(self) -> float:
        live = ~self.dead_slots if len(self.dead_slots) else np.zeros(0, bool)
        tot = float(self.doc_len[live].sum()) if len(self.doc_len) else 0.0
        tot += float(sum(sum(d.values()) for d in self.delta.values())) if len(self.delta) < 50_000 else 0.0
        n = max(1, self.n_docs)
        return max(1.0, tot / n)

    # ---- building ----
    def build(self, ids: np.ndarray, term_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
              doc_len: np.ndarray) -> None:
        """term_rows: chunks of (term_hash uint64, doc_slot int32, tf uint16),
        doc slots index `ids` (which must be sorted ascending)."""
        with self.lock:
            if term_rows:
                th = np.concatenate([r[0] for r in term_rows])
                ds = np.concatenate([r[1] for r in term_rows])
                tf = np.concatenate([r[2] for r in term_rows])
            else:
                th = np.zeros(0, np.uint64)
                ds = np.zeros(0, np.int32)
                tf = np.zeros(0, np.uint16)
            order = np.lexsort((ds, th))
            th, ds, tf = th[order], ds[order], tf[order]
            uniq, start = np.unique(th, return_index=True)
            self.term_hash = uniq.astype(np.uint64)
            self.offs = np.append(start, len(th)).astype(np.int64)
            self.docs = ds.astype(np.int32)
            self.tfs = tf.astype(np.uint16)
            self.doc_ids = np.asarray(ids, dtype=np.int64)
            self.doc_len = np.asarray(doc_len, dtype=np.int32)
            self.dead_slots = np.zeros(len(self.doc_ids), dtype=bool)
            self.delta.clear()
            self.delta_post.clear()
            self.dead_ids.clear()
            self.dirty = True

    def compact(self) -> None:
        """Fold delta + tombstones into a new base."""
        with self.lock:
            if not self.delta and not self.dead_slots.any():
                return
            keep = ~self.dead_slots
            new_slot = np.cumsum(keep) - 1
            pmask = keep[self.docs] if len(self.docs) else np.zeros(0, bool)
            # posting term per entry
            counts = np.diff(self.offs)
            th = np.repeat(self.term_hash, counts)[pmask]
            ds = new_slot[self.docs[pmask]].astype(np.int64)
            tf = self.tfs[pmask]
            ids = list(self.doc_ids[keep].tolist())
            lens = list(self.doc_len[keep].tolist())
            extra_th, extra_ds, extra_tf = [], [], []
            for i, terms in sorted(self.delta.items()):
                slot = len(ids)
                ids.append(i)
                lens.append(sum(terms.values()))
                for t, c in terms.items():
                    extra_th.append(term_hash(t))
                    extra_ds.append(slot)
                    extra_tf.append(min(c, 65535))
            ids_a = np.asarray(ids, dtype=np.int64)
            order = np.argsort(ids_a, kind="stable")
            remap = np.empty(len(order), dtype=np.int64)
            remap[order] = np.arange(len(order))
            all_th = np.concatenate([th, np.asarray(extra_th, np.uint64)])
            all_ds = remap[np.concatenate([ds, np.asarray(extra_ds, np.int64)])].astype(np.int32)
            all_tf = np.concatenate([tf, np.asarray(extra_tf, np.uint16)])
            self.build(ids_a[order], [(all_th, all_ds, all_tf)], np.asarray(lens, np.int32)[order])

    # ---- incremental ----
    def remove(self, ids: Iterable[int]) -> None:
        with self.lock:
            ids = [int(i) for i in ids]
            if not ids:
                return
            for i in ids:
                terms = self.delta.pop(i, None)
                if terms is not None:
                    for t in terms:
                        p = self.delta_post.get(t)
                        if p is not None:
                            p.pop(i, None)
                            if not p:
                                del self.delta_post[t]
            if len(self.doc_ids):
                q = np.asarray(ids, dtype=np.int64)
                pos = np.searchsorted(self.doc_ids, q)
                pos = np.clip(pos, 0, len(self.doc_ids) - 1)
                hit = self.doc_ids[pos] == q
                if hit.any():
                    self.dead_slots[pos[hit]] = True
                    self.dead_ids.update(q[hit].tolist())
            self.dirty = True

    def add(self, docs: Iterable[tuple[int, Counter]]) -> None:
        with self.lock:
            for i, terms in docs:
                i = int(i)
                d = {t: int(c) for t, c in terms.items()}
                self.delta[i] = d
                for t, c in d.items():
                    self.delta_post[t][i] = c
            self.dirty = True

    # ---- query ----
    def topk(self, q_tokens: list[str], k: int) -> list[tuple[int, float]]:
        toks = list(dict.fromkeys(t for t in q_tokens if t))
        if not toks:
            return []
        with self.lock:
            n = max(1, self.n_docs)
            avgdl = self._avgdl()
            nb = len(self.doc_ids)
            scores = np.zeros(nb, dtype=np.float32) if nb else None
            dscore: dict[int, float] = defaultdict(float)
            for t in toks:
                h = np.uint64(term_hash(t))
                df = 0
                sl = None
                if nb:
                    j = int(np.searchsorted(self.term_hash, h))
                    if j < len(self.term_hash) and self.term_hash[j] == h:
                        a, b = int(self.offs[j]), int(self.offs[j + 1])
                        sl = (a, b)
                        df += b - a
                dp = self.delta_post.get(t)
                if dp:
                    df += len(dp)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                if sl is not None:
                    a, b = sl
                    ds = self.docs[a:b]
                    tf = self.tfs[a:b].astype(np.float32)
                    dl = self.doc_len[ds].astype(np.float32)
                    contrib = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
                    np.add.at(scores, ds, contrib)
                if dp:
                    for i, tf in dp.items():
                        dl = sum(self.delta[i].values())
                        dscore[i] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
            out: list[tuple[int, float]] = []
            if nb:
                if self.dead_slots.any():
                    scores[self.dead_slots] = 0.0
                nz = int(np.count_nonzero(scores))
                if nz:
                    kk = min(k, nz)
                    top = np.argpartition(-scores, kk - 1)[:kk]
                    out = [(int(self.doc_ids[s]), float(scores[s])) for s in top if scores[s] > 0]
            out += list(dscore.items())
            out.sort(key=lambda x: (-x[1], x[0]))
            return out[:k]

    # ---- persistence ----
    def save(self, base: bool = False) -> None:
        import orjson
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            if base:
                p = self.dir / f"{self.label}.npz"
                tmp = self.dir / f"{self.label}.npz.tmp"
                with open(tmp, "wb") as fh:
                    np.savez(fh, term_hash=self.term_hash, offs=self.offs, docs=self.docs, tfs=self.tfs,
                             doc_ids=self.doc_ids, doc_len=self.doc_len)
                os.replace(tmp, p)
            dp = self.dir / f"{self.label}.delta.json"
            tmp = self.dir / f"{self.label}.delta.json.tmp"
            tmp.write_bytes(orjson.dumps({"delta": {str(i): d for i, d in self.delta.items()},
                                          "dead": sorted(self.dead_ids)}))
            os.replace(tmp, dp)
            self.dirty = False

    def load(self) -> bool:
        import orjson
        p = self.dir / f"{self.label}.npz"
        if not p.exists():
            return False
        with self.lock:
            try:
                with np.load(p, allow_pickle=False) as z:
                    self.term_hash = z["term_hash"]
                    self.offs = z["offs"]
                    self.docs = z["docs"]
                    self.tfs = z["tfs"]
                    self.doc_ids = z["doc_ids"]
                    self.doc_len = z["doc_len"]
            except Exception:
                self._empty_base()
                return False
            self.dead_slots = np.zeros(len(self.doc_ids), dtype=bool)
            self.delta.clear()
            self.delta_post.clear()
            self.dead_ids.clear()
            dp = self.dir / f"{self.label}.delta.json"
            if dp.exists():
                try:
                    d = orjson.loads(dp.read_bytes())
                    dead = d.get("dead") or []
                    if dead:
                        self.remove(dead)
                    self.add((int(i), Counter(t)) for i, t in (d.get("delta") or {}).items())
                except Exception:
                    pass
            self.dirty = False
            return True


class KeywordIndex:
    """The three per-label indexes of one root."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "kw"
        self.labels = {lb: LabelIndex(lb, self.dir) for lb in LABELS}
        self.loaded = False

    def load(self) -> bool:
        ok = all([self.labels[lb].load() for lb in LABELS])
        self.loaded = ok
        return ok

    def available(self) -> bool:
        return self.loaded

    def topk(self, label: str, q_tokens: list[str], k: int) -> list[tuple[int, float]]:
        return self.labels[label].topk(q_tokens, k)

    def remove(self, label: str, ids: Iterable[int]) -> None:
        self.labels[label].remove(ids)

    def add(self, label: str, docs: Iterable[tuple[int, Counter]]) -> None:
        self.labels[label].add(docs)

    def delta_size(self) -> int:
        return sum(len(x.delta) + len(x.dead_ids) for x in self.labels.values())

    def base_size(self) -> int:
        return sum(len(x.doc_ids) for x in self.labels.values())

    def save(self, base: bool = False) -> None:
        for x in self.labels.values():
            if base or x.dirty:
                x.save(base=base)
        self.loaded = True

    def compact(self) -> None:
        for x in self.labels.values():
            x.compact()
        self.save(base=True)


class KeywordBuilder:
    """Streaming builder used by a full index pass: add docs batch by batch,
    finish() writes the base."""

    def __init__(self) -> None:
        self.parts: dict[str, list] = {lb: [] for lb in LABELS}
        self.ids: dict[str, list[np.ndarray]] = {lb: [] for lb in LABELS}
        self.lens: dict[str, list[np.ndarray]] = {lb: [] for lb in LABELS}
        self.count: dict[str, int] = {lb: 0 for lb in LABELS}

    def add(self, label: str, docs: list[tuple[int, Counter]]) -> None:
        if not docs:
            return
        base = self.count[label]
        th, ds, tf = [], [], []
        ids, lens = [], []
        for j, (i, terms) in enumerate(docs):
            ids.append(int(i))
            lens.append(sum(terms.values()))
            for t, c in terms.items():
                th.append(term_hash(t))
                ds.append(base + j)
                tf.append(min(int(c), 65535))
        self.parts[label].append((np.asarray(th, np.uint64), np.asarray(ds, np.int64), np.asarray(tf, np.uint16)))
        self.ids[label].append(np.asarray(ids, np.int64))
        self.lens[label].append(np.asarray(lens, np.int32))
        self.count[label] += len(docs)

    def finish(self, kw: KeywordIndex) -> None:
        for lb in LABELS:
            ids = np.concatenate(self.ids[lb]) if self.ids[lb] else np.zeros(0, np.int64)
            lens = np.concatenate(self.lens[lb]) if self.lens[lb] else np.zeros(0, np.int32)
            order = np.argsort(ids, kind="stable")
            remap = np.empty(len(order), dtype=np.int64)
            remap[order] = np.arange(len(order))
            parts = [(th, remap[ds].astype(np.int32), tf) for th, ds, tf in self.parts[lb]]
            kw.labels[lb].build(ids[order], parts, lens[order])
        kw.save(base=True)
