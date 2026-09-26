"""Hybrid retrieval: vector + name match + graph expansion + PageRank rerank.

All Cypher queries here are Kuzu-flavored:
  - `label(x)` works for both nodes and relationships
  - no `type(r)`, `startNode()`, `endNode()`, `relationships(path)`
"""
from __future__ import annotations

import re

import numpy as np

from docgraph.bm25 import BM25Index, rrf_fuse, tokenize
from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.git_tools import blame_lines, changed_entities, recent_commits
from docgraph.rank import PersonalizedRanker
from docgraph.rerank import Reranker
from docgraph.rules import rules_for as _rules_for


from concurrent.futures import ThreadPoolExecutor as _TPE

# Shared by every Retriever: search fans its index legs out over it.
_SEARCH_POOL = _TPE(max_workers=6, thread_name_prefix="docgraph-search")
# Second level (inside a leg: row fetch || embedding fetch) -- a separate pool
# so a leg never waits on a worker of its own pool.
_SEARCH_POOL2 = _TPE(max_workers=8, thread_name_prefix="docgraph-search2")


class Retriever:
    def __init__(self, db: GraphDB, embedder: Embedder, cfg: Config | None = None,
                 workspace=None):
        self.db = db
        self.embedder = embedder
        self.cfg = cfg
        # When set, the reranker is borrowed from the workspace's pool
        # rather than created locally. This keeps the cross-encoder
        # session shared across roots AND tracked by the workspace's
        # idle unloader — without a workspace ref we'd hold a private
        # Reranker that would never be eligible for eviction.
        self.workspace = workspace
        self._ranker: PersonalizedRanker | None = None
        self._reranker: Reranker | None = None
        # Per-label BM25 indexes built on first use. Keyed by label so each
        # search() call only touches the relevant corpus.
        self._bm25: dict[str, tuple[BM25Index, list[int]]] = {}
        # Keyword index (kwindex.KeywordIndex). The host hands in the root's
        # live one (kept current by the indexer in-process); otherwise it is
        # loaded from `.docgraph/kw/` on first use and reloaded when the
        # files change.
        self.kw = None
        self._kw_stamp: tuple | None = None
        # pending_vectors(label, id) -> vector | None: vectors the host has
        # deferred (not in the vector index yet), for scoring fresh entities
        self.pending_vectors = None

    def _kw_index(self):
        from docgraph.kwindex import KeywordIndex
        if self.kw is not None and getattr(self, "_kw_owned", False) is False:
            return self.kw
        if self.cfg is None:
            return None
        d = self.cfg.data_dir / "kw"
        try:
            stamp = tuple(sorted((f.name, f.stat().st_mtime_ns) for f in d.iterdir()))
        except OSError:
            return None
        if self.kw is None or stamp != self._kw_stamp:
            kw = KeywordIndex(self.cfg.data_dir)
            if not kw.load():
                return None
            self.kw, self._kw_stamp, self._kw_owned = kw, stamp, True
        return self.kw

    def _kw_topk(self, label: str, toks: list[str], k: int) -> list[tuple[int, float]]:
        kw = self._kw_index()
        if kw is None or not toks:
            return []
        return kw.topk(label, toks, k)

    def _reranker_(self) -> Reranker:
        # Prefer the workspace pool so the idle unloader can see it.
        if self.workspace is not None and self.cfg is not None:
            return self.workspace.reranker_for(self.cfg)
        if self._reranker is None:
            # cfg.rerank_model may be "" — Reranker falls back to its built-in
            # default (jinaai/jina-reranker-v1-tiny-en).
            model = getattr(self.cfg, "rerank_model", "") or None
            from .embed import resolve_device
            self._reranker = Reranker(
                model_name=model,
                device=resolve_device(getattr(self.cfg, "rerank_gpu", False)),
                torch_compile=getattr(self.cfg, "rerank_torch_compile", False),
            )
        return self._reranker

    def _ranker_(self) -> PersonalizedRanker:
        if self._ranker is None:
            self._ranker = PersonalizedRanker(self.db)
        return self._ranker

    def _bm25_for(self, label: str, rows: list[dict]) -> tuple[BM25Index, list[int]] | None:
        """Build (or fetch cached) BM25 index for a label's corpus. The index
        scores `name + body` per row. We cache by label so the first search hit
        pays the build cost (~tokenize + posting build) once."""
        cached = self._bm25.get(label)
        if cached is not None and len(cached[1]) == len(rows):
            return cached
        if not rows:
            return None
        docs: list[list[str]] = []
        ids: list[int] = []
        for r in rows:
            text = f"{r.get('name','')} {r.get('qname','')} {r.get('body') or ''}"
            docs.append(tokenize(text))
            ids.append(r["id"])
        idx = BM25Index(docs)
        self._bm25[label] = (idx, ids)
        return self._bm25[label]

    def _chunk_max_sims(self, qvec) -> dict[str, float]:
        """For each parent_qname, the best cosine similarity across its
        sub-chunks. Empty when no chunks exist."""
        try:
            meta = self.db.fetch_all("MATCH (c:Chunk) RETURN c.id AS id, c.parent_qname AS qname")
            if not meta:
                return {}
            vid, vmat = self.db.embedding_matrix("Chunk")
        except Exception:
            return {}
        pos = {int(i): j for j, i in enumerate(vid.tolist())}
        rows = [r for r in meta if int(r["id"]) in pos]
        if not rows:
            return {}
        mat = vmat[[pos[int(r["id"])] for r in rows]]
        qv = np.array(qvec, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        sims = (mat @ qv).tolist()
        out: dict[str, float] = {}
        for r, s in zip(rows, sims):
            q = r["qname"]
            if q not in out or s > out[q]:
                out[q] = float(s)
        return out

    def _redact(self, file: str | None, body: str | None, snippet: str | None = None) -> tuple[str | None, str | None]:
        """Mask body/snippet if the file is AI-blocked. Returns (body, snippet)."""
        if not file or self.cfg is None:
            return body, snippet
        if self.cfg.ai_blocked_logical(file):
            return "[redacted by .cursorignore]", "[redacted]"
        return body, snippet

    # ---- search ----------------------------------------------------------
    #
    # Indexed path (schema v4+): candidates come from Kuzu's HNSW vector
    # index and the keyword index (kwindex.py; schema v5 -- Kuzu's FTS crashes
    # on incremental deletes) per label (+ an exact-name probe and the parents
    # of the best-matching sub-chunks), so a query touches O(k) rows, never
    # the whole table. Scoring (cosine + RRF of the vector / keyword ranks + name +
    # PageRank / PPR boosts, optional cross-encoder rerank) runs on that
    # candidate set only. Pre-v4 databases (no indexes) use the brute-force
    # path (`_search_brute`) unchanged.

    SEARCH_K = 60

    def search_backend(self) -> str:
        """"index" when every vector index exists, else "brute"."""
        cached = getattr(self, "_search_backend_cache", None)
        if cached is None:
            have = self.db.list_indexes()
            need = list(self.db.VECTOR_INDEXES.values())
            cached = "index" if all(n in have for n in need) else "brute"
            self._search_backend_cache = cached
        return cached

    def search(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 10,
        focus_file: str | None = None,
        focus_symbol: str | None = None,
        rerank: bool = False,
    ) -> list[dict]:
        """Hybrid search. If focus_file or focus_symbol is provided, ranks
        results by personalized PageRank biased toward that focus point --
        the model sees results most relevant to where the agent is working.

        rerank=True runs a cross-encoder over the top candidates for
        token-level precision (downloads a small ~33 MB model on first use).
        Plain-text / symbol-less files come back as `label: "File"` hits
        carrying their best chunk's line + snippet. kind: function | class |
        file (files only) | None (everything).
        """
        if self.search_backend() != "index":
            return self._search_brute(query, kind=kind, limit=limit, focus_file=focus_file,
                                      focus_symbol=focus_symbol, rerank=rerank)
        eq = getattr(self.embedder, "embed_query", None)
        qvec = eq(query) if callable(eq) else self.embedder.embed([query])[0]
        qv = self._unit(qvec)
        k = max(self.SEARCH_K, int(limit) * 5)
        want_files = kind in (None, "", "file", "text")
        if kind in ("file", "text"):
            labels: tuple[str, ...] = ()
        elif kind == "function":
            labels = ("Function",)
        elif kind == "class":
            labels = ("Class",)
        else:
            labels = ("Function", "Class")
        ppr = self._maybe_ppr(focus_file, focus_symbol)
        q_tokens = tokenize(query)
        words = re.findall(r"[A-Za-z0-9_]+", query)
        kw_toks = list(dict.fromkeys(q_tokens + [w.lower() for w in words]))
        qlow = query.lower().strip()
        results: list[dict] = []

        # -- index legs, run concurrently on per-thread connections --------
        ident = bool(qlow) and not any(ch.isspace() for ch in qlow)
        legs: dict[str, object] = {
            "c_vec": lambda: self._safe(lambda: self.db.vector_topk("Chunk", qv, k * 2), []),
            "c_kw": lambda: self._safe(lambda: self._kw_topk("Chunk", kw_toks, k), []),
        }
        for label in labels:
            legs[f"{label}_vec"] = (lambda lb: lambda: self._safe(lambda: self.db.vector_topk(lb, qv, k), []))(label)
            legs[f"{label}_kw"] = (lambda lb: lambda: self._safe(lambda: self._kw_topk(lb, kw_toks, k), []))(label)
            if ident:   # a multi-word query can never equal a symbol name
                legs[f"{label}_name"] = (lambda lb: lambda: [r["id"] for r in self._safe(lambda: self.db.fetch_all(
                    f"MATCH (n:{lb}) WHERE lower(n.name) = $q RETURN n.id AS id LIMIT 50",
                    {"q": qlow}), [])])(label)
        got = self._parallel(legs)
        # -- sub-chunk leg: best chunk similarity per parent --------------
        chunk_vec = got["c_vec"]
        chunk_kw = got["c_kw"]
        cids = list({i for i, _ in chunk_vec} | {i for i, _ in chunk_kw})
        crow: dict[int, dict] = {}
        if cids:
            for r in self.db.fetch_all(
                "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.parent_qname AS pq, "
                "c.parent_label AS pl, c.file AS file, c.line_start AS line, "
                "substring(c.body, 1, 400) AS body", {"ids": cids}):
                crow[r["id"]] = r
        chunk_sim: dict[int, float] = {i: s_ for i, s_ in chunk_vec}
        need_c = [i for i in crow if i not in chunk_sim]
        if need_c:
            eids, emat = self._safe(lambda: self.db.embeddings_for("Chunk", need_c),
                                    (np.zeros(0, np.int64), np.zeros((0, 1), np.float32)))
            if len(eids):
                en = emat / (np.linalg.norm(emat, axis=1, keepdims=True) + 1e-9)
                for i, s_ in zip(eids.tolist(), (en @ qv).tolist()):
                    chunk_sim[i] = float(s_)
        for cid in crow:
            chunk_sim.setdefault(cid, 0.0)
        chunk_max: dict[str, float] = {}
        for cid, r in crow.items():
            if r.get("pl") in ("Function", "Class"):
                q = r.get("pq")
                if q and chunk_sim[cid] > chunk_max.get(q, -1.0):
                    chunk_max[q] = chunk_sim[cid]
        chunk_kw_rank = {cid: i for i, (cid, _s) in enumerate(chunk_kw)}

        def score_label(label: str) -> list[dict]:
            out: list[dict] = []
            vec = got[f"{label}_vec"]
            kw = got[f"{label}_kw"]
            ids = {i for i, _ in vec} | {i for i, _ in kw}
            ids |= set(got.get(f"{label}_name") or ())
            parents = [q for q in chunk_max if q]
            if not ids and not parents:
                return out
            # cosine of the vector hits comes from the index; only keyword /
            # name / chunk-parent candidates need their embedding read. The
            # row scan (ids + chunk parents in ONE pass) and the embedding
            # read run concurrently.
            sim_of = {i: s_ for i, s_ in vec}
            need0 = [i for i in ids if i not in sim_of]
            db = self.db
            cols = ("n.id AS id, n.name AS name, n.qname AS qname, n.file AS file, "
                    "n.line_start AS line_start, substring(n.body, 1, 400) AS body, "
                    "n.pagerank AS pagerank, n.llm_doc AS llm_doc")

            def fetch_rows():
                with db.thread_conn():
                    if parents:
                        return db.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids OR n.qname IN $qs "
                                            f"RETURN {cols}", {"ids": list(ids), "qs": parents})
                    return db.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN {cols}",
                                        {"ids": list(ids)})

            def fetch_emb(want):
                with db.thread_conn():
                    got = self._safe(lambda: db.embeddings_for(label, want),
                                     (np.zeros(0, np.int64), np.zeros((0, 1), np.float32)))
                # entities whose vectors the host has not written yet
                pend = self.pending_vectors
                if pend is not None:
                    have = set(got[0].tolist())
                    extra = [(i, v) for i in want if i not in have for v in [pend(label, i)] if v is not None]
                    if extra:
                        ids2 = np.array([i for i, _ in extra], dtype=np.int64)
                        mat2 = np.array([v for _, v in extra], dtype=np.float32)
                        if len(got[0]):
                            return np.concatenate([got[0], ids2]), np.concatenate([got[1], mat2])
                        return ids2, mat2
                return got
            f_rows = _SEARCH_POOL2.submit(fetch_rows)
            f_emb = _SEARCH_POOL2.submit(fetch_emb, need0) if need0 else None
            rows = f_rows.result()
            if not rows:
                return out
            parts = [f_emb.result()] if f_emb is not None else []
            late = [r["id"] for r in rows if r["id"] not in sim_of and r["id"] not in ids]
            if late:                                    # chunk-parent rows
                parts.append(fetch_emb(late))
            for eids, emat in parts:
                if len(eids):
                    en = emat / (np.linalg.norm(emat, axis=1, keepdims=True) + 1e-9)
                    for i, s_ in zip(eids.tolist(), (en @ qv).tolist()):
                        sim_of[i] = float(s_)
            sims = [float(sim_of.get(r["id"], 0.0)) for r in rows]
            best = [max(s_, chunk_max.get(r["qname"], -1.0)) for r, s_ in zip(rows, sims)]
            idx_of = {r["id"]: i for i, r in enumerate(rows)}
            vec_order = sorted(range(len(rows)), key=lambda i: best[i], reverse=True)
            kw_order = [idx_of[i] for i, _s in kw if i in idx_of]
            fused = rrf_fuse(vec_order, kw_order)
            for i, r in enumerate(rows):
                name_boost = 0.3 if qlow and qlow in (r["name"] or "").lower() else 0.0
                pr = r.get("pagerank") or 0.0
                ppr_boost = ppr.get(r["id"], 0.0) if ppr else 0.0
                rank_term = (ppr_boost * 0.5) if ppr else (pr * 0.1)
                score = best[i] + name_boost + rank_term + float(fused.get(i, 0.0)) * 8.0
                _, snippet = self._redact(r["file"], None, (r["body"] or "")[:300])
                out.append({
                    "label": label, "id": r["id"], "name": r["name"], "qname": r["qname"],
                    "file": r["file"], "line": r["line_start"], "snippet": snippet,
                    "llm_doc": r.get("llm_doc"), "score": float(score), "pagerank": float(pr),
                    "ppr": float(ppr_boost),
                })
            return out

        legs2 = {lb: (lambda lb_: lambda: score_label(lb_))(lb) for lb in labels}
        # -- file-level hits (plain text, docs, configs, notebook markdown) --
        if want_files:
            legs2["_files"] = lambda: self._file_hits(crow, chunk_sim, chunk_kw, chunk_kw_rank, qlow, ppr)
        for part in self._parallel(legs2).values():
            results.extend(part)

        results.sort(key=lambda x: x["score"], reverse=True)
        if rerank and results:
            try:
                results = self._reranker_().rerank(query, results[:max(50, limit)],
                                                   text_key="snippet", top_k=50)
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(f"Rerank failed, falling back: {e}")
        return results[:limit]

    def _file_hits(self, crow: dict[int, dict], chunk_sim: dict[int, float],
                   chunk_kw: list[tuple[int, float]], chunk_kw_rank: dict[int, int],
                   qlow: str, ppr: dict | None) -> list[dict]:
        best_chunk: dict[str, int] = {}
        for cid, r in crow.items():
            if r.get("pl") != "File" or not r.get("file"):
                continue
            f = r["file"]
            if f not in best_chunk or chunk_sim[cid] > chunk_sim[best_chunk[f]]:
                best_chunk[f] = cid
        if not best_chunk:
            return []
        files = list(best_chunk)
        frows = {r["path"]: r for r in self.db.fetch_all(
            "MATCH (f:File) WHERE f.path IN $p RETURN f.id AS id, f.path AS path, "
            "coalesce(f.pagerank, 0.0) AS pr", {"p": files})}
        order = sorted(files, key=lambda f: chunk_sim[best_chunk[f]], reverse=True)
        kw_files: list[str] = []
        kw_best: dict[str, int] = {}
        for cid, _s in chunk_kw:
            r = crow.get(cid) or {}
            f = r.get("file")
            if r.get("pl") == "File" and f in best_chunk:
                if f not in kw_best:
                    kw_best[f] = cid
                    kw_files.append(f)
        fused = rrf_fuse(order, kw_files)
        out: list[dict] = []
        for f in files:
            fr = frows.get(f)
            if fr is None:
                continue
            cid = best_chunk[f]
            show = kw_best.get(f, cid)
            base = f.rsplit("/", 1)[-1].lower()
            name_boost = 0.3 if qlow and qlow in base else 0.0
            pr = float(fr.get("pr") or 0.0)
            ppr_boost = ppr.get(fr["id"], 0.0) if ppr else 0.0
            rank_term = (ppr_boost * 0.5) if ppr else (pr * 0.1)
            score = chunk_sim[cid] + name_boost + rank_term + float(fused.get(f, 0.0)) * 8.0
            _, snippet = self._redact(f, None, (crow[show].get("body") or "")[:300])
            out.append({
                "label": "File", "id": fr["id"], "name": f, "qname": f, "file": f,
                "line": crow[show].get("line") or 1, "snippet": snippet, "llm_doc": None,
                "score": float(score), "pagerank": pr, "ppr": float(ppr_boost),
            })
        return out

    def _parallel(self, legs: dict) -> dict:
        """Run independent read legs concurrently, each on its own thread's
        connection (GraphDB.thread_conn); falls back to sequential."""
        if len(legs) <= 1:
            return {k: fn() for k, fn in legs.items()}

        def run(fn):
            with self.db.thread_conn():
                return fn()
        try:
            futs = {k: _SEARCH_POOL.submit(run, fn) for k, fn in legs.items()}
            return {k: f.result() for k, f in futs.items()}
        except Exception:  # noqa: BLE001 - e.g. DB closed under us: plain path
            return {k: fn() for k, fn in legs.items()}

    @staticmethod
    def _unit(v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        return a / (np.linalg.norm(a) + 1e-9)

    @staticmethod
    def _safe(fn, default):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - an index hiccup degrades one leg only
            return default

    def _search_brute(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 10,
        focus_file: str | None = None,
        focus_symbol: str | None = None,
        rerank: bool = False,
    ) -> list[dict]:
        """Hybrid search. If focus_file or focus_symbol is provided, ranks
        results by personalized PageRank biased toward that focus point —
        the model sees results most relevant to where the agent is working.

        rerank=True runs a cross-encoder over the top candidates for
        token-level precision (downloads a small ~33 MB model on first use).
        """
        eq = getattr(self.embedder, "embed_query", None)
        qvec = eq(query) if callable(eq) else self.embedder.embed([query])[0]
        results: list[dict] = []
        labels = ("Function",) if kind == "function" else ("Class",) if kind == "class" else ("Function", "Class")

        ppr = self._maybe_ppr(focus_file, focus_symbol)

        # Per-entity max chunk similarity (sub-function chunking lift):
        # for any qname, the best score across its sub-chunks rivals the
        # entity-level score so a query that matches a small piece of a
        # 500-line function still surfaces it.
        chunk_max = self._chunk_max_sims(qvec)

        # Tokenize the query once for the BM25 leg; if every token is too short
        # to clear BM25Index's min length, we silently skip the keyword fuse.
        q_tokens = tokenize(query)
        qlow = query.lower()

        for label in labels:
            rows = self.db.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                f"n.file AS file, n.line_start AS line_start, n.body AS body, "
                f"n.pagerank AS pagerank, n.llm_doc AS llm_doc"
            )
            if not rows:
                continue
            vid, vmat = self.db.embedding_matrix(label)
            pos = {int(i): j for j, i in enumerate(vid.tolist())}
            rows = [r for r in rows if int(r["id"]) in pos]
            if not rows:
                continue
            mat = vmat[[pos[int(r["id"])] for r in rows]]
            qv = np.array(qvec, dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            sims = mat @ qv

            # Best chunk sim per qname → take max(entity_sim, best_chunk_sim)
            # so long-body entities don't lose recall when only a section matches.
            best_sims = [max(float(s), chunk_max.get(r["qname"], -1.0))
                         for r, s in zip(rows, sims.tolist())]

            # BM25 keyword score over name+qname+body. Fused with vector via RRF
            # (k=60, classic Cormack constant). If BM25 is empty (no tokens
            # match), the fused rank degenerates back to pure vector — same
            # behavior as before this change.
            bm25_pair = self._bm25_for(label, rows)
            bm25_scores: list[float] = []
            if bm25_pair and q_tokens:
                idx, _id_list = bm25_pair
                bm25_scores = idx.score(q_tokens)
            else:
                bm25_scores = [0.0] * len(rows)

            # Build rank lists (descending). Indices into `rows`.
            n = len(rows)
            vec_order = sorted(range(n), key=lambda i: best_sims[i], reverse=True)
            kw_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
            # Drop trailing zero-BM25 entries — they shouldn't earn rank credit.
            kw_order = [i for i in kw_order if bm25_scores[i] > 0.0]
            fused = rrf_fuse(vec_order, kw_order)

            for i, r in enumerate(rows):
                name_boost = 0.3 if qlow in r["name"].lower() else 0.0
                pr = r.get("pagerank") or 0.0
                ppr_boost = ppr.get(r["id"], 0.0) if ppr else 0.0
                rank_term = (ppr_boost * 0.5) if ppr else (pr * 0.1)
                # Combine: vector best_sim (semantic anchor) + RRF fusion bonus
                # + name match + PR. RRF scores are tiny (<0.05) — multiplied
                # so a strong dual ranking wins ties between similarly-scored
                # vector candidates without overwhelming a clear semantic match.
                rrf_bonus = float(fused.get(i, 0.0)) * 8.0
                score = best_sims[i] + name_boost + rank_term + rrf_bonus
                _, snippet = self._redact(r["file"], None, (r["body"] or "")[:300])
                results.append({
                    "label": label,
                    "id": r["id"],
                    "name": r["name"],
                    "qname": r["qname"],
                    "file": r["file"],
                    "line": r["line_start"],
                    "snippet": snippet,
                    "llm_doc": r.get("llm_doc"),
                    "score": float(score),
                    "pagerank": float(pr),
                    "ppr": float(ppr_boost),
                })
        results.sort(key=lambda x: x["score"], reverse=True)

        if rerank and results:
            try:
                results = self._reranker_().rerank(
                    query, results, text_key="snippet", top_k=50,
                )
            except Exception as e:  # noqa: BLE001
                # Don't fail the search if the reranker can't load (offline,
                # no model, etc.) — degrade silently to bi-encoder ranking.
                import logging
                logging.getLogger(__name__).warning(f"Rerank failed, falling back: {e}")
        return results[:limit]

    def _focus_ids(self, focus_file: str | None, focus_symbol: str | None) -> list[int]:
        """Translate a file path or symbol name into seed node IDs."""
        ids: list[int] = []
        if focus_file:
            for label, prop in (("File", "path"), ("Function", "file"), ("Class", "file")):
                for r in self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.{prop} = $f RETURN n.id AS id",
                    {"f": focus_file},
                ):
                    ids.append(r["id"])
        if focus_symbol:
            for label in ("Function", "Class"):
                for r in self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.name = $s RETURN n.id AS id",
                    {"s": focus_symbol},
                ):
                    ids.append(r["id"])
        return ids

    def _maybe_ppr(
        self, focus_file: str | None, focus_symbol: str | None
    ) -> dict[int, float] | None:
        if not focus_file and not focus_symbol:
            return None
        ids = self._focus_ids(focus_file, focus_symbol)
        if not ids:
            return None
        try:
            return self._ranker_().personalized(ids)
        except Exception:
            return None

    def definition(self, name: str, file: str | None = None) -> list[dict]:
        params: dict = {"name": name}
        where = "n.name = $name"
        if file:
            where += " AND n.file = $file"
            params["file"] = file
        rows = []
        for label in ("Function", "Class"):
            for r in self.db.fetch_all(
                f"MATCH (n:{label}) WHERE {where} "
                f"RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                f"n.file AS file, n.line_start AS line, n.body AS body, n.llm_doc AS llm_doc",
                params,
            ):
                r["label"] = label
                body, _ = self._redact(r.get("file"), r.get("body"))
                r["body"] = body
                rows.append(r)
        return rows

    def references(self, name: str) -> list[dict]:
        out: list[dict] = []
        for edge in ("CALLS", "REFERENCES_", "INSTANTIATES"):
            try:
                rows = self.db.fetch_all(
                    f"MATCH (target)<-[r:{edge}]-(src) WHERE target.name = $name "
                    f"RETURN src.qname AS caller, src.name AS caller_name, src.file AS file, "
                    f"src.line_start AS line, label(src) AS caller_kind",
                    {"name": name},
                )
                for r in rows:
                    r["edge"] = edge
                    out.append(r)
            except Exception:
                pass
        return out

    # --- confidence-aware CALLS traversal ---------------------------------
    #
    # Multi-hop walks run as a Python BFS over 1-hop queries (frontier IN
    # $ids) rather than Cypher var-length patterns: it lets every hop filter
    # on `r.confidence`, and it never touches the backward var-length +
    # `nodes(path)` shape that segfaults Kuzu 0.11 (see CLAUDE.md).

    def _cap(self, key: str) -> bool:
        caps = getattr(self, "_caps_cache", None)
        if caps is None:
            caps = {
                "calls_conf": "confidence" in self.db.table_props("CALLS"),
                "community": self.db.has_table("Community"),
                "routes": self.db.has_table("Route"),
                "tools": self.db.has_table("Tool"),
                "cand": self.db.has_table("CALLS_CANDIDATE"),
                "history": "last_changed_commit" in self.db.table_props("Function"),
            }
            self._caps_cache = caps
        return bool(caps.get(key, False))

    def _call_edges(self, ids, direction: str = "out", min_conf: float = 0.0,
                    rel: str = "CALLS") -> list[dict]:
        """1-hop CALLS rows touching `ids`: {src, dst, conf, method, line}."""
        ids = list(ids)
        if not ids:
            return []
        has_conf = self._cap("calls_conf") if rel == "CALLS" else self._cap("cand")
        conf = "coalesce(r.confidence, 1.0)" if has_conf else "1.0"
        meth = "coalesce(r.method, '')" if has_conf else "''"
        side = "a" if direction == "out" else "b"
        where = f"{side}.id IN $ids"
        params: dict = {"ids": ids}
        if min_conf > 0 and has_conf:
            where += f" AND {conf} >= $c"
            params["c"] = float(min_conf)
        try:
            return self.db.fetch_all(
                f"MATCH (a:Function)-[r:{rel}]->(b:Function) WHERE {where} "
                f"RETURN a.id AS src, b.id AS dst, {conf} AS conf, {meth} AS method, "
                f"r.line AS line",
                params,
            )
        except Exception:
            return []

    def _bfs_calls(self, seeds, direction: str, depth: int, min_conf: float = 0.0,
                   max_nodes: int = 20000, with_edges: bool = False
                   ) -> tuple[dict[int, int], dict[int, int], list[dict]]:
        """BFS over CALLS. Returns (dist, parent, edges): dist[id] = hops
        from the nearest seed, parent[id] = predecessor on that path.

        Runs on the in-memory CSR of `_mem_graph()` (built once per index
        generation, ~0.3 s on 2M edges), a level per numpy pass: the old
        one-query-per-hop walk cost ~5 ms per hop plus the row decoding, and
        tools like processes() walk hundreds of hops. Ties (several parents
        at the same depth) go to the smallest parent id, so results are
        deterministic. `edges` (with line / method) is only fetched when
        asked for: the CALLS rows among the reached nodes, one query."""
        mg = self._mem_graph()
        adj = mg["fwd"] if direction == "out" else mg["rev"]
        seed_arr = np.unique(np.asarray([int(x) for x in seeds], dtype=np.int64))
        dist: dict[int, int] = {int(x): 0 for x in seed_arr.tolist()}
        parent: dict[int, int] = {}
        seen_sorted = seed_arr
        frontier = seed_arr
        for d in range(1, max(0, depth) + 1):
            if not len(frontier):
                break
            owner, nb, cf = adj.expand(frontier)
            if min_conf > 0 and len(nb):
                keep = cf >= min_conf
                owner, nb = owner[keep], nb[keep]
            if not len(nb):
                break
            pos = np.searchsorted(seen_sorted, nb)
            pos[pos >= len(seen_sorted)] = 0
            fresh = seen_sorted[pos] != nb
            owner, nb = owner[fresh], nb[fresh]
            if not len(nb):
                break
            order = np.lexsort((owner, nb))
            nb, owner = nb[order], owner[order]
            first = np.ones(len(nb), dtype=bool)
            first[1:] = nb[1:] != nb[:-1]
            nb, owner = nb[first], owner[first]
            for x, p in zip(nb.tolist(), owner.tolist()):
                dist[x] = d
                parent[x] = p
            seen_sorted = np.union1d(seen_sorted, nb)
            frontier = nb
            if len(dist) > max_nodes:
                break
        edges: list[dict] = []
        if with_edges and len(dist) > 1:
            edges = self._call_edges_among(list(dist), min_conf)
        return dist, parent, edges

    def _call_edges_among(self, ids, min_conf: float = 0.0) -> list[dict]:
        """CALLS rows with both ends in `ids`: {src, dst, conf, method, line}."""
        ids = list(ids)
        if not ids:
            return []
        has_conf = self._cap("calls_conf")
        conf = "coalesce(r.confidence, 1.0)" if has_conf else "1.0"
        meth = "coalesce(r.method, '')" if has_conf else "''"
        where = "a.id IN $ids AND b.id IN $ids"
        params: dict = {"ids": ids}
        if min_conf > 0 and has_conf:
            where += f" AND {conf} >= $c"
            params["c"] = float(min_conf)
        try:
            return self.db.fetch_all(
                f"MATCH (a:Function)-[r:CALLS]->(b:Function) WHERE {where} "
                f"RETURN a.id AS src, b.id AS dst, {conf} AS conf, {meth} AS method, "
                f"r.line AS line ORDER BY src, dst, line", params)
        except Exception:
            return []

    # `_nodes` answers from the in-memory payload table (built once per
    # generation, ~40 ms / ~30 MB for 220k symbols, prewarmed by the host):
    # Kuzu's `id IN $list` costs ~27 us per listed id (10k ids = 0.3 s) plus
    # a ~15 ms floor per label. Above this many ids the table is built on
    # demand; below it the Cypher path is used until the table exists.
    NODES_TABLE_MIN = 0

    def _node_table(self):
        """(sorted ids, Arrow table) of every Function / Class payload,
        built once per retriever (= per index generation), under a lock."""
        g = getattr(self, "_node_table_cache", None)
        if g is not None:
            return g
        with self._build_lock:
            g = getattr(self, "_node_table_cache", None)
            if g is not None:
                return g
            import pyarrow as pa
            parts = []
            for label in ("Function", "Class"):
                test_col = "n.is_test" if label == "Function" else "false"
                t = self.db.fetch_arrow(
                    f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                    f"n.file AS file, n.line_start AS line, n.line_end AS line_end, "
                    f"CAST(coalesce(n.pagerank, 0.0) AS DOUBLE) AS pagerank, {test_col} AS is_test, "
                    f"'{label}' AS kind")
                parts.append(t.cast(pa.schema([
                    ("id", pa.int64()), ("name", pa.string()), ("qname", pa.string()),
                    ("file", pa.string()), ("line", pa.int64()), ("line_end", pa.int64()),
                    ("pagerank", pa.float64()), ("is_test", pa.bool_()), ("kind", pa.string())])))
            tbl = pa.concat_tables(parts).combine_chunks()
            ids = np.asarray(tbl.column("id").to_numpy(zero_copy_only=False), dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            tbl = tbl.take(pa.array(order))
            g = (ids[order], tbl)
            self._node_table_cache = g
            return g

    @property
    def _build_lock(self):
        lk = self.__dict__.get("_build_lock_obj")
        if lk is None:
            import threading
            lk = self.__dict__.setdefault("_build_lock_obj", threading.RLock())
        return lk

    def _nodes(self, ids) -> dict[int, dict]:
        """Function/Class payloads by id."""
        ids = [int(i) for i in ids]
        out: dict[int, dict] = {}
        if not ids:
            return out
        if len(ids) >= self.NODES_TABLE_MIN or getattr(self, "_node_table_cache", None) is not None:
            try:
                import pyarrow as pa
                tids, tbl = self._node_table()
                q = np.unique(np.asarray(ids, dtype=np.int64))
                pos = np.searchsorted(tids, q)
                pos[pos >= len(tids)] = 0
                hit = pos[tids[pos] == q] if len(tids) else pos[:0]
                for r in tbl.take(pa.array(hit)).to_pylist():
                    out[r["id"]] = r
                return out
            except Exception:
                out = {}
        for label in ("Function", "Class"):
            test_col = "n.is_test" if label == "Function" else "false"
            try:
                rows = self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.id IN $ids "
                    f"RETURN n.id AS id, n.name AS name, n.qname AS qname, n.file AS file, "
                    f"n.line_start AS line, n.line_end AS line_end, "
                    f"coalesce(n.pagerank, 0.0) AS pagerank, {test_col} AS is_test",
                    {"ids": ids},
                )
            except Exception:
                rows = []
            for r in rows:
                r["kind"] = label
                out[r["id"]] = r
        return out

    def _symbol_ids(self, name: str, labels=("Function",), file: str | None = None) -> list[int]:
        """Ids for a symbol given by name or qname (optionally file-scoped)."""
        ids: list[int] = []
        if not name:
            return ids
        key = "qname" if "::" in name else "name"
        for label in labels:
            where = f"n.{key} = $n" + (" AND n.file = $f" if file else "")
            params = {"n": name, **({"f": file} if file else {})}
            try:
                for r in self.db.fetch_all(f"MATCH (n:{label}) WHERE {where} RETURN n.id AS id", params):
                    ids.append(r["id"])
            except Exception:
                pass
        return ids

    def call_graph(self, name: str, depth: int = 2, min_confidence: float = 0.0) -> dict:
        depth = max(1, min(depth, 5))
        seeds = self._symbol_ids(name)
        if not seeds:
            return {"calls": [], "called_by": [], "edges": []}
        db = self.db

        def conn(fn, *a):
            def run():
                with db.thread_conn():
                    return fn(*a)
            return run
        # both walks and the 1-hop edge rows are independent: concurrently
        f_out = _SEARCH_POOL2.submit(conn(self._bfs_calls, seeds, "out", depth, min_confidence))
        f_in = _SEARCH_POOL2.submit(conn(self._bfs_calls, seeds, "in", depth, min_confidence))
        f_eo = _SEARCH_POOL2.submit(conn(self._call_edges, seeds, "out", min_confidence))
        f_ei = _SEARCH_POOL2.submit(conn(self._call_edges, seeds, "in", min_confidence))
        fdist, _fp, _fe = f_out.result()
        bdist, _bp, _be = f_in.result()
        nodes = self._nodes(set(fdist) | set(bdist))

        def row(i: int, d: int) -> dict:
            n = nodes.get(i, {})
            return {"qname": n.get("qname"), "name": n.get("name"), "file": n.get("file"),
                    "line": n.get("line"), "depth": d}

        forward_rows = [row(i, d) for i, d in sorted(fdist.items(), key=lambda t: (t[1], t[0])) if i in nodes]
        backward_rows = [row(i, d) for i, d in sorted(bdist.items(), key=lambda t: (t[1], t[0]))
                         if i in nodes and d > 0]
        # Direct edges (1 hop) for an actual edge list, with confidence
        edges = []
        seen: set[tuple] = set()
        for r in f_eo.result() + f_ei.result():
            key = (r["src"], r["dst"])
            if key in seen:
                continue
            seen.add(key)
            a, b = nodes.get(r["src"]), nodes.get(r["dst"])
            if not a or not b:
                extra = self._nodes([r["src"], r["dst"]])
                a, b = extra.get(r["src"]), extra.get(r["dst"])
            if a and b:
                edges.append({"src": a["qname"], "dst": b["qname"],
                              "confidence": r["conf"], "method": r["method"]})
        return {"calls": forward_rows, "called_by": backward_rows, "edges": edges}

    def file_map(self, file: str) -> dict:
        entities: list[dict] = []
        for label in ("Function", "Class"):
            for r in self.db.fetch_all(
                f"MATCH (n:{label}) WHERE n.file = $file "
                f"RETURN n.name AS name, n.qname AS qname, n.line_start AS line, "
                f"n.pagerank AS pagerank ORDER BY n.line_start",
                {"file": file},
            ):
                r["kind"] = label
                entities.append(r)
        entities.sort(key=lambda x: x["line"])
        imports_file = self.db.fetch_all(
            "MATCH (f:File)-[:IMPORTS]->(m:File) WHERE f.path = $file "
            "RETURN m.path AS target, 'File' AS kind",
            {"file": file},
        )
        imports_mod = self.db.fetch_all(
            "MATCH (f:File)-[:IMPORTS]->(m:Module) WHERE f.path = $file "
            "RETURN m.name AS target, 'Module' AS kind",
            {"file": file},
        )
        return {"entities": entities, "imports": imports_file + imports_mod}

    def neighborhood(self, name: str, limit: int = 10) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, int]] = set()
        db = self.db
        EDGES = ("CALLS", "REFERENCES_", "SIMILAR_TO", "INHERITS", "TESTS")

        def q(edge: str):
            try:
                with db.thread_conn():
                    return db.fetch_all(
                        f"MATCH (n)-[r:{edge}]-(other) WHERE n.name = $name AND other.name IS NOT NULL "
                        f"RETURN DISTINCT other.qname AS qname, other.name AS name, other.file AS file, "
                        f"other.line_start AS line, label(other) AS kind, "
                        f"coalesce(other.pagerank, 0.0) AS pagerank",
                        {"name": name},
                    )
            except Exception:
                return None
        # the edge types are independent: one connection each, concurrently
        futs = [_SEARCH_POOL2.submit(q, edge) for edge in EDGES]
        for edge, f in zip(EDGES, futs):
            rows = f.result()
            if rows is None:
                continue
            if True:
                for r in rows:
                    key = (r["qname"], 0)
                    if key in seen:
                        continue
                    seen.add(key)
                    r["via"] = edge
                    out.append(r)
        out.sort(key=lambda x: x.get("pagerank") or 0.0, reverse=True)
        return out[:limit]

    # --- Multi-hop / impact / test_impact / cypher ---------------------

    def node_neighbors(self, node_id: int, hops: int = 1) -> dict:
        """Lazy-fetch a node's neighborhood for UI expansion. Returns the
        node's 1..N-hop neighbors as `{nodes, edges}` in the same shape as
        `graph_dump`, ready to be merged into an existing canvas.

        Used by the UI's level-of-detail expansion and focus-depth lazy load,
        so the graph never depends on what happened to fit in the initial
        10k-node dump."""
        hops = max(1, min(int(hops), 5))
        # Locate the seed (it may be any label; just use raw id match).
        try:
            seed_rows = self.db.fetch_all(
                "MATCH (n) WHERE n.id = $id "
                "RETURN n.id AS id, label(n) AS kind, "
                "coalesce(n.name, n.path) AS name, "
                "coalesce(n.file, n.path) AS file, "
                "coalesce(n.pagerank, 0.0) AS pagerank",
                {"id": int(node_id)},
            )
        except Exception:
            return {"nodes": [], "edges": []}
        if not seed_rows:
            return {"nodes": [], "edges": []}

        seen_ids: set[int] = {int(node_id)}
        edge_records: list[dict] = []
        frontier: set[int] = {int(node_id)}

        EDGE_TYPES = (
            "CONTAINS", "CALLS", "IMPORTS", "IMPORTS_SYMBOL", "INHERITS",
            "IMPLEMENTS", "OVERRIDES", "REFERENCES_", "INSTANTIATES",
            "DECORATED_BY", "RETURNS", "SIMILAR_TO", "TESTS", "CO_CHANGED_WITH",
            "LINKS_TO",
        )
        for _ in range(hops):
            if not frontier:
                break
            next_front: set[int] = set()
            ids = list(frontier)
            # One directed query per (edge type, side), all run concurrently
            # on per-thread connections (Kuzu executes reads of different
            # connections in parallel): out of / into the frontier, so each
            # edge keeps its real direction.
            db = self.db

            def q(edge, side, _ids=ids):
                if not db.has_table(edge):
                    return edge, []
                try:
                    with db.thread_conn():
                        return edge, db.fetch_all(
                            f"MATCH (a)-[r:{edge}]->(b) WHERE {side}.id IN $ids "
                            f"AND a.id IS NOT NULL AND b.id IS NOT NULL "
                            f"RETURN a.id AS src, b.id AS dst", {"ids": _ids})
                except Exception:
                    return edge, []
            futs = [_SEARCH_POOL2.submit(q, edge, side) for edge in EDGE_TYPES for side in ("a", "b")]
            for f in futs:
                edge, rows = f.result()
                for row in rows:
                    src, dst = int(row["src"]), int(row["dst"])
                    edge_records.append({"src": src, "dst": dst, "kind": edge})
                    if dst not in seen_ids:
                        next_front.add(dst)
                        seen_ids.add(dst)
                    if src not in seen_ids:
                        next_front.add(src)
                        seen_ids.add(src)
            frontier = next_front

        # Resolve all collected ids to node payloads. We re-coalesce because
        # File uses `path` while everything else uses `name`/`file`.
        all_ids = list(seen_ids)
        try:
            rows = self.db.fetch_all(
                "MATCH (n) WHERE n.id IN $ids "
                "RETURN n.id AS id, label(n) AS kind, "
                "coalesce(n.name, n.path) AS name, "
                "coalesce(n.file, n.path) AS file, "
                "coalesce(n.pagerank, 0.0) AS pagerank",
                {"ids": all_ids},
            )
        except Exception:
            rows = []
        nodes = [r for r in rows if r.get("name")]
        # De-duplicate edges so the merge step doesn't double-draw lines.
        seen_edge: set[tuple[int, int, str]] = set()
        deduped: list[dict] = []
        for e in edge_records:
            key = (e["src"], e["dst"], e["kind"])
            if key in seen_edge:
                continue
            seen_edge.add(key)
            deduped.append(e)
        self._annotate_graph(nodes, deduped)
        return {"nodes": nodes, "edges": deduped}

    def explore(
        self,
        seeds: list[str],
        hops: int = 3,
        limit: int = 25,
        edges: tuple[str, ...] = ("CALLS", "REFERENCES_", "SIMILAR_TO", "INHERITS", "TESTS"),
        min_confidence: float = 0.0,
    ) -> dict:
        """Multi-hop graph walk from one or more seed names. Returns nodes
        ranked by min-distance and pagerank — the agent gets a 1-shot view of
        the relevant subgraph instead of having to chain `neighborhood` calls.

        seeds: symbol names (Function/Class). hops: 1..5. min_confidence
        drops CALLS / INHERITS hops below that resolution confidence.
        """
        hops = max(1, min(int(hops), 5))
        if not seeds:
            return {"nodes": [], "edges": []}

        # Resolve seed names to IDs (Function or Class)
        seed_ids: list[int] = []
        by_name: dict[str, list[int]] = {}
        for label in ("Function", "Class"):
            for r in self.db.fetch_all(f"MATCH (n:{label}) WHERE n.name IN $s RETURN n.id AS id, n.name AS name",
                                       {"s": list(dict.fromkeys(seeds))}):
                by_name.setdefault(r["name"], []).append(r["id"])
        for s in seeds:
            seed_ids.extend(by_name.get(s, ()))
        if not seed_ids:
            return {"nodes": [], "edges": []}

        # BFS on per-edge-type undirected CSRs held in memory (built once per
        # retriever = per index generation): an IN-list hop query over a
        # 10^4-node frontier cost ~0.3 s per edge type and level.
        conf_types = ("CALLS", "INHERITS", "INSTANTIATES")
        adjs = [(k, self._explore_adj(edge)) for k, edge in enumerate(edges)]
        seed_arr = np.unique(np.asarray(seed_ids, dtype=np.int64))
        seen_ids = [seed_arr]
        seen_dist = [np.zeros(len(seed_arr), dtype=np.int16)]
        seen_sorted = seed_arr
        frontier = seed_arr
        e_src: list[np.ndarray] = []
        e_dst: list[np.ndarray] = []
        e_kind: list[np.ndarray] = []
        for d in range(1, hops + 1):
            if not len(frontier):
                break
            dsts_all = []
            for k, adj in adjs:
                if adj is None:
                    continue
                src_a, dst_a, cf = adj.expand(frontier)
                if min_confidence > 0 and edges[k] in conf_types and len(src_a):
                    keep = cf >= min_confidence
                    src_a, dst_a = src_a[keep], dst_a[keep]
                if len(src_a):
                    e_src.append(src_a)
                    e_dst.append(dst_a)
                    e_kind.append(np.full(len(src_a), k, dtype=np.int8))
                    dsts_all.append(dst_a)
            if not dsts_all:
                break
            cand = np.unique(np.concatenate(dsts_all))
            pos = np.searchsorted(seen_sorted, cand)
            pos[pos >= len(seen_sorted)] = 0
            fresh = cand[seen_sorted[pos] != cand]
            seen_ids.append(fresh)
            seen_dist.append(np.full(len(fresh), d, dtype=np.int16))
            seen_sorted = np.union1d(seen_sorted, fresh)
            frontier = fresh

        all_ids = np.concatenate(seen_ids)
        all_dist = np.concatenate(seen_dist)
        # Rank every reached Function / Class from an in-memory (id, pagerank)
        # table, then fetch the payload of the top `limit` only (an IN over
        # 10^5 reached ids cost seconds).
        pr_ids, pr_val = self._symbol_pagerank()
        pos = np.searchsorted(pr_ids, all_ids)
        pos[pos >= len(pr_ids)] = 0
        is_sym = (pr_ids[pos] == all_ids) if len(pr_ids) else np.zeros(len(all_ids), bool)
        sym_ids, sym_d = all_ids[is_sym], all_dist[is_sym]
        sym_pr = pr_val[pos[is_sym]] if len(pr_ids) else np.zeros(0)
        score = 1.0 / (sym_d.astype(np.float64) + 1.0) + sym_pr * 0.5
        top = np.lexsort((sym_ids, -score))[:max(0, int(limit))]
        top_ids = sym_ids[top].tolist()
        info = self._nodes(top_ids)
        nodes = []
        for i, d, sc in zip(top_ids, sym_d[top].tolist(), score[top].tolist()):
            n = info.get(i)
            if not n:
                continue
            nodes.append({"id": i, "name": n["name"], "qname": n["qname"], "file": n["file"],
                          "line": n["line"], "kind": n["kind"], "pagerank": n["pagerank"],
                          "distance": d, "score": sc})
        # Edges of the returned subgraph (both ends among the returned nodes),
        # deduplicated; the walk's full edge count is reported alongside.
        edge_records: list[dict] = []
        total = 0
        if e_src:
            src_c = np.concatenate(e_src)
            dst_c = np.concatenate(e_dst)
            kind_c = np.concatenate(e_kind)
            total = int(len(src_c))
            keep_ids = np.asarray(sorted(n["id"] for n in nodes), dtype=np.int64)
            m = np.isin(src_c, keep_ids) & np.isin(dst_c, keep_ids)
            seen_e: set = set()
            for a_, b_, k_ in zip(src_c[m].tolist(), dst_c[m].tolist(), kind_c[m].tolist()):
                key = (a_, b_, k_)
                if key in seen_e:
                    continue
                seen_e.add(key)
                edge_records.append({"src": a_, "dst": b_, "kind": edges[k_]})
        return {"nodes": nodes, "edges": edge_records, "edges_walked": total,
                "reached": int(len(all_ids))}

    def _explore_adj(self, rel: str):
        """Undirected CSR over one rel table (both directions), cached."""
        cache = getattr(self, "_explore_adj_cache", None)
        if cache is None:
            cache = self._explore_adj_cache = {}
        if rel in cache:
            return cache[rel]
        adj = None
        if self.db.has_table(rel):
            with_conf = "confidence" in self.db.table_props(rel)
            a, b, c, _k = self.db.edge_endpoints((rel,), with_conf=with_conf)
            if c is None:
                c = np.ones(len(a), dtype=np.float32)
            adj = _CSRAdj(np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([c, c]))
        cache[rel] = adj
        return adj

    def _symbol_pagerank(self):
        """(sorted ids, pagerank) of every Function and Class, cached."""
        g = getattr(self, "_symbol_pr_cache", None)
        if g is not None:
            return g
        ids, prs = [], []
        for label in ("Function", "Class"):
            d = self.db.node_columns(label, {"id": "n.id", "p": "coalesce(n.pagerank, 0.0)"})
            if d.get("id") is not None and len(d["id"]):
                ids.append(np.asarray(d["id"], dtype=np.int64))
                prs.append(np.asarray(d["p"], dtype=np.float64))
        if ids:
            i = np.concatenate(ids)
            p = np.concatenate(prs)
            o = np.argsort(i, kind="stable")
            g = (i[o], p[o])
        else:
            g = (np.zeros(0, np.int64), np.zeros(0, np.float64))
        self._symbol_pr_cache = g
        return g

    def _callers_rows(self, seed_ids: list[int], depth: int, limit: int,
                      min_conf: float) -> list[dict]:
        dist, _p, _e = self._bfs_calls(seed_ids, "in", depth, min_conf)
        callers = [i for i, d in dist.items() if d > 0]
        nodes = self._nodes(callers)
        rows = [{"qname": n["qname"], "name": n["name"], "file": n["file"], "line": n["line"],
                 "pagerank": n["pagerank"], "depth": dist[i]} for i, n in nodes.items()]
        rows.sort(key=lambda r: (-(r["pagerank"] or 0.0), r["depth"]))
        return rows[:limit]

    def impact_of(
        self,
        target: str,
        depth: int = 3,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> dict:
        if min_confidence > 0:
            return self._impact_of_conf(target, depth, limit, min_confidence)
        return self._impact_of_legacy(target, depth, limit)

    def _impact_of_conf(self, target: str, depth: int, limit: int, min_conf: float) -> dict:
        """impact_of with a confidence floor on every CALLS hop."""
        out = self._impact_of_legacy(target, depth, limit)
        depth = max(1, min(int(depth), 5))
        is_file = bool(self.db.fetch_all(
            "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}))
        if is_file:
            seeds = [r["id"] for r in self.db.fetch_all(
                "MATCH (n:Function) WHERE n.file = $t RETURN n.id AS id", {"t": target})]
        else:
            seeds = self._symbol_ids(target)
        out["callers"] = self._callers_rows(seeds, depth, limit, min_conf)
        out["min_confidence"] = min_conf
        return out

    def _impact_of_legacy(
        self,
        target: str,
        depth: int = 3,
        limit: int = 50,
    ) -> dict:
        """Blast radius of a file or symbol. Returns:
          - callers: transitive callers (CALLS reverse, up to `depth` hops)
          - importers: files that import this file
          - co_changed: files that historically changed alongside
          - tests: tests that exercise the target

        target: a symbol name OR a file path. We try file first, then symbol.
        """
        depth = max(1, min(int(depth), 5))
        out: dict = {"target": target, "callers": [], "importers": [], "co_changed": [], "tests": []}

        is_file = bool(self.db.fetch_all(
            "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}
        ))

        if is_file:
            # Importers
            out["importers"] = self.db.fetch_all(
                "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE b.path = $t "
                "RETURN a.path AS file",
                {"t": target},
            )
            # Co-changed
            out["co_changed"] = self.db.fetch_all(
                "MATCH (a:File)-[r:CO_CHANGED_WITH]-(b:File) WHERE a.path = $t "
                "RETURN b.path AS file, r.count AS count ORDER BY r.count DESC LIMIT 25",
                {"t": target},
            )
            # Transitive callers of any function in this file
            try:
                rows = self.db.fetch_all(
                    # Endpoint form -- see call_graph (Kuzu nodes(path) segfault).
                    f"MATCH (caller:Function)-[:CALLS*1..{depth}]->(callee:Function) "
                    f"WHERE callee.file = $t "
                    f"RETURN DISTINCT caller.qname AS qname, caller.name AS name, "
                    f"caller.file AS file, caller.line_start AS line, "
                    f"coalesce(caller.pagerank,0.0) AS pagerank "
                    f"ORDER BY pagerank DESC LIMIT $lim",
                    {"t": target, "lim": limit},
                )
                out["callers"] = rows
            except Exception:
                out["callers"] = []
            # Tests
            try:
                out["tests"] = self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.file = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["tests"] = []
        else:
            # Symbol path
            try:
                out["callers"] = self.db.fetch_all(
                    # Endpoint form -- see call_graph (Kuzu nodes(path) segfault).
                    f"MATCH (caller)-[:CALLS*1..{depth}]->(target:Function) "
                    f"WHERE target.name = $t "
                    f"RETURN DISTINCT caller.qname AS qname, caller.name AS name, "
                    f"caller.file AS file, caller.line_start AS line, "
                    f"coalesce(caller.pagerank,0.0) AS pagerank "
                    f"ORDER BY pagerank DESC LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["callers"] = []
            try:
                out["tests"] = self.db.fetch_all(
                    "MATCH (test:Function)-[:TESTS]->(target) WHERE target.name = $t "
                    "RETURN test.name AS name, test.file AS file, test.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["tests"] = []
            # File of the symbol → its importers + co-changed
            files_of_symbol = self.db.fetch_all(
                "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.name = $t "
                "RETURN DISTINCT n.file AS file LIMIT 5",
                {"t": target},
            )
            for fr in files_of_symbol:
                f = fr["file"]
                out["importers"].extend(self.db.fetch_all(
                    "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE b.path = $f RETURN a.path AS file",
                    {"f": f},
                ))
                out["co_changed"].extend(self.db.fetch_all(
                    "MATCH (a:File)-[r:CO_CHANGED_WITH]-(b:File) WHERE a.path = $f "
                    "RETURN b.path AS file, r.count AS count ORDER BY r.count DESC LIMIT 10",
                    {"f": f},
                ))
        return out

    def _entry_points(self, limit: int) -> list[dict]:
        """Flow starts: route / MCP-tool handlers first, then Functions with
        no incoming CALLS, by PageRank."""
        out: list[dict] = []
        seen: set[int] = set()
        for label in ("Route", "Tool"):
            if not self._cap("routes" if label == "Route" else "tools"):
                continue
            try:
                rows = self.db.fetch_all(
                    f"MATCH (r:{label})-[:HANDLES]->(f:Function) "
                    f"RETURN f.id AS id, f.qname AS qname, f.name AS name, f.file AS file, "
                    f"f.line_start AS line, coalesce(f.pagerank, 0.0) AS pagerank, "
                    f"r.name AS via_name, r.id AS via_id "
                    f"ORDER BY r.file, r.line",
                )
            except Exception:
                rows = []
            for r in rows:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                r["kind"] = "route" if label == "Route" else "tool"
                out.append(r)
        rows = self.db.fetch_all(
            "MATCH (f:Function) WHERE NOT EXISTS { MATCH ()-[:CALLS]->(f) } "
            "AND coalesce(f.pagerank, 0.0) > 0.0 AND NOT coalesce(f.is_test, false) "
            "RETURN f.id AS id, f.qname AS qname, f.name AS name, f.file AS file, "
            "f.line_start AS line, coalesce(f.pagerank, 0.0) AS pagerank "
            "ORDER BY pagerank DESC LIMIT $lim",
            {"lim": int(limit) * 3},
        )
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            r["kind"] = "entry"
            r["via_name"] = None
            out.append(r)
        return out

    def _flow_walk(self, entry: dict, max_chain_len: int, min_confidence: float = 0.0,
                   max_nodes: int = 50):
        dist, parent, _e = self._bfs_calls([entry["id"]], "out", max_chain_len,
                                           min_confidence, max_nodes=max_nodes * 4)
        ids = [i for i, d in sorted(dist.items(), key=lambda t: (t[1], t[0])) if d > 0][:max_nodes]
        return ids, dist, parent

    def _flow_build(self, entry: dict, ids, dist, parent, nodes: dict, edges: list[dict]) -> dict | None:
        if not ids:
            return None
        keep = set(ids) | {entry["id"]}
        chain = []
        for i in ids:
            n = nodes.get(i)
            if not n:
                continue
            p = nodes.get(parent.get(i))
            chain.append({"id": i, "qname": n["qname"], "name": n["name"], "file": n["file"],
                          "line": n["line"], "depth": dist[i],
                          "parent": p["qname"] if p else None})
        call_edges = []
        seen: set[tuple] = set()
        for r in edges:
            if r["src"] in keep and r["dst"] in keep and (r["src"], r["dst"]) not in seen:
                seen.add((r["src"], r["dst"]))
                a, b = nodes.get(r["src"]), nodes.get(r["dst"])
                if a and b:
                    call_edges.append({"src": a["qname"], "dst": b["qname"],
                                       "line": r.get("line") or 0,
                                       "confidence": r.get("conf"), "method": r.get("method")})
        return {
            "id": entry["id"],
            "kind": entry.get("kind", "entry"),
            "via": entry.get("via_name"),
            "entry": {"id": entry["id"], "qname": entry["qname"], "name": entry["name"],
                      "file": entry["file"], "line": entry["line"], "pagerank": entry["pagerank"]},
            "chain": chain,
            "edges": call_edges,
            "fanout": len(chain),
        }

    def _flows(self, entries: list[dict], max_chain_len: int, min_confidence: float,
               max_nodes: int, limit: int) -> list[dict]:
        """Flows for `entries` (up to `limit` non-empty ones): in-memory walks,
        then one node fetch and one edge fetch for all of them."""
        walks = []
        for e in entries:
            ids, dist, parent = self._flow_walk(e, max_chain_len, min_confidence, max_nodes)
            if ids:
                walks.append((e, ids, dist, parent))
                if len(walks) >= limit:
                    break
        if not walks:
            return []
        every: set[int] = set()
        for e, ids, _d, _p in walks:
            every.update(ids)
            every.add(e["id"])
        nodes = self._nodes(sorted(every))
        edges = self._call_edges_among(sorted(every), min_confidence)
        out = []
        for e, ids, dist, parent in walks:
            flow = self._flow_build(e, ids, dist, parent, nodes, edges)
            if flow:
                out.append(flow)
        return out

    def _flow_for(self, entry: dict, max_chain_len: int, min_confidence: float = 0.0,
                  max_nodes: int = 50) -> dict | None:
        flows = self._flows([entry], max_chain_len, min_confidence, max_nodes, 1)
        return flows[0] if flows else None

    def processes(self, limit: int = 25, max_chain_len: int = 8,
                  min_confidence: float = 0.0) -> list[dict]:
        """Detect 'processes' = top-level execution flows. Entry points are
        route / MCP-tool handlers (framework maps) and Functions with no
        incoming CALLS, by PageRank. For each, walk forward through CALLS up
        to `max_chain_len` hops (BFS, confidence-filtered) and return the
        chain plus the call edges between its members (the UI draws them as
        a sequence diagram)."""
        max_chain_len = max(2, min(int(max_chain_len), 12))
        key = (int(limit), max_chain_len, float(min_confidence))
        memo = getattr(self, "_processes_memo", None)
        if memo is None:
            memo = self._processes_memo = {}
        if key in memo:            # per Retriever = per index generation
            return memo[key]
        if len(memo) > 16:
            memo.clear()
        out = self._flows(self._entry_points(limit), max_chain_len, min_confidence, 50, int(limit))
        memo[key] = out
        return out

    def flow(self, entry_id: int, max_chain_len: int = 8, min_confidence: float = 0.0) -> dict | None:
        """One flow by its entry Function id (the `id` of a processes() row)."""
        n = self._nodes([int(entry_id)]).get(int(entry_id))
        if not n:
            return None
        entry = dict(n, kind="entry", via_name=None)
        for label, kind in (("Route", "route"), ("Tool", "tool")):
            if not self._cap("routes" if label == "Route" else "tools"):
                continue
            rows = self.db.fetch_all(
                f"MATCH (r:{label})-[:HANDLES]->(f:Function) WHERE f.id = $id RETURN r.name AS n LIMIT 1",
                {"id": int(entry_id)})
            if rows:
                entry.update(kind=kind, via_name=rows[0]["n"])
                break
        return self._flow_for(entry, max(2, min(int(max_chain_len), 12)), min_confidence, max_nodes=120)

    def test_impact(self, target: str, limit: int = 25, min_confidence: float = 0.0) -> list[dict]:
        if min_confidence > 0:
            base = [r for r in self._test_impact_legacy(target, limit) if r.get("via") == "TESTS"]
            is_file = bool(self.db.fetch_all(
                "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}))
            seeds = ([r["id"] for r in self.db.fetch_all(
                "MATCH (n:Function) WHERE n.file = $t RETURN n.id AS id", {"t": target})]
                if is_file else self._symbol_ids(target))
            dist, _p, _e = self._bfs_calls(seeds, "in", 3, min_confidence)
            seen = {(r.get("name"), r.get("file")) for r in base}
            for n in self._nodes([i for i, d in dist.items() if d > 0]).values():
                if n.get("is_test") and (n["name"], n["file"]) not in seen:
                    seen.add((n["name"], n["file"]))
                    base.append({"name": n["name"], "file": n["file"], "line": n["line"], "via": "CALLS*"})
            return base[:limit]
        return self._test_impact_legacy(target, limit)

    def _test_impact_legacy(self, target: str, limit: int = 25) -> list[dict]:
        """Tests that exercise `target` (file or symbol). Differentiator:
        we already have TESTS edges + reverse CALLS, no competitor exposes
        this as a primitive."""
        is_file = bool(self.db.fetch_all(
            "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}
        ))
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []

        def add(rows: list[dict], via: str) -> None:
            for r in rows:
                key = (r.get("name"), r.get("file"))
                if key in seen:
                    continue
                seen.add(key)
                r["via"] = via
                out.append(r)
                if len(out) >= limit:
                    return

        if is_file:
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.file = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line",
                    {"t": target},
                ), "TESTS")
            except Exception:
                pass
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:CALLS*1..3]->(callee:Function) "
                    "WHERE callee.file = $t AND t.is_test = true "
                    "RETURN DISTINCT t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                ), "CALLS*")
            except Exception:
                pass
        else:
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.name = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line",
                    {"t": target},
                ), "TESTS")
            except Exception:
                pass
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:CALLS*1..3]->(callee:Function) "
                    "WHERE callee.name = $t AND t.is_test = true "
                    "RETURN DISTINCT t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                ), "CALLS*")
            except Exception:
                pass
        return out[:limit]

    # Cypher escape hatch ---------------------------------------------------

    _WRITE_KEYWORDS = (
        "CREATE", "MERGE", "DELETE", "DETACH",
        "SET", "REMOVE", "DROP", "ALTER", "COPY",
    )

    @classmethod
    def _is_read_only(cls, query: str) -> bool:
        upper = query.upper()
        # Strip string literals so e.g. "MERGE" inside a string doesn't false-positive
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", upper)
        for kw in cls._WRITE_KEYWORDS:
            if re.search(rf"\b{kw}\b", stripped):
                return False
        return True

    def cypher(self, query: str, limit: int = 100) -> dict:
        """Read-only Cypher escape hatch. Lets the agent author its own graph
        queries — none of the competitors expose this. Rejects writes; caps
        rows.

        Returns {"rows": [...], "rejected": str | None}.
        """
        if not self._is_read_only(query):
            return {"rows": [], "rejected": "write keyword detected (CREATE/MERGE/SET/DELETE/...)"}
        # Append a LIMIT safety net unless one exists
        if "LIMIT" not in query.upper():
            query = f"{query.rstrip(';')} LIMIT {int(limit)}"
        try:
            return {"rows": self.db.fetch_all(query)[:limit], "rejected": None}
        except Exception as e:  # noqa: BLE001
            return {"rows": [], "rejected": f"query error: {e}"}

    # --- Git-aware retrieval ----------------------------------------------

    def git_changes(self, ref: str | None = None) -> dict:
        """Diff-aware retrieval. ref:
          - None    → unstaged + staged working-tree diff
          - "HEAD"  → last commit
          - "main"  → branch diff vs main
          - "<sha>" → that commit

        Returns changed files + entities + the 1-hop callers of changed
        functions, so the agent gets a 'what's about to break' picture in one
        call. Mirrors Cursor's @Commit / @Recent Changes / @PR but joined to
        the graph.
        """
        if self.cfg is None:
            return {"ref": ref, "files": [], "entities": [], "callers_of_changed": [],
                    "error": "Config not attached to Retriever"}
        return changed_entities(self.cfg, self.db, ref)

    def git_blame(self, file: str, line_start: int = 1, line_end: int | None = None) -> list[dict]:
        """`git blame` for a file/line range. Mirrors Cursor Blame."""
        if self.cfg is None:
            return []
        return blame_lines(self.cfg, file, line_start=line_start, line_end=line_end)

    def git_recent(self, file: str | None = None, limit: int = 20) -> list[dict]:
        """Recent commits, optionally scoped to a file path."""
        if self.cfg is None:
            return []
        return recent_commits(self.cfg, file_path=file, limit=limit)

    # --- Auto-attach rules (.cursor/rules/*.mdc + AGENTS.md / CLAUDE.md) --

    def rules_for(self, file: str) -> list[dict]:
        """Cursor-rules-compatible auto-attach: return rules whose globs
        match `file`, plus AGENTS.md / CLAUDE.md as always-apply."""
        if self.cfg is None:
            return []
        return _rules_for(self.cfg, file)

    _ALL_GRAPH_EDGES = (
        "CONTAINS", "CALLS", "IMPORTS", "IMPORTS_SYMBOL", "INHERITS",
        "IMPLEMENTS", "OVERRIDES", "REFERENCES_", "INSTANTIATES",
        "DECORATED_BY", "RETURNS", "SIMILAR_TO", "TESTS", "CO_CHANGED_WITH",
        "LINKS_TO",
    )

    def graph_dump(self, limit_nodes: int = 2000) -> dict:
        """Top-`limit_nodes` subgraph by PageRank (legacy API; the UI streams
        tiles instead). Memoized per Retriever, i.e. until the next reindex."""
        memo = getattr(self, "_graph_dump_memo", None)
        if memo is None:
            memo = self._graph_dump_memo = {}
        key = int(limit_nodes)
        if key not in memo:
            if len(memo) > 4:
                memo.clear()
            memo[key] = self._graph_dump(key)
        return memo[key]

    def _graph_dump(self, limit_nodes: int = 2000) -> dict:
        # Top-K by PageRank, proportional across labels. Functions get the
        # biggest slice (most numerous + structurally interesting); Files,
        # Classes, Variables share the rest. A 200k-symbol codebase still
        # respects limit_nodes — no label can drown out the others.
        # Shares: Function 50%, Class 20%, File 20%, Variable 10%.
        budgets = {
            "Function": max(1, limit_nodes // 2),
            "Class":    max(1, limit_nodes // 5),
            "File":     max(1, limit_nodes // 5),
            "Variable": max(1, limit_nodes // 10),
        }
        nodes: list[dict] = []
        for label, lim in budgets.items():
            try:
                if label == "File":
                    rows = self.db.fetch_all(
                        f"MATCH (n:File) RETURN n.id AS id, n.path AS name, n.path AS file, "
                        f"coalesce(n.pagerank, 0.0) AS pagerank "
                        f"ORDER BY pagerank DESC LIMIT {lim}"
                    )
                else:
                    rows = self.db.fetch_all(
                        f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.file AS file, "
                        f"coalesce(n.pagerank, 0.0) AS pagerank "
                        f"ORDER BY pagerank DESC LIMIT {lim}"
                    )
            except Exception:
                rows = []
            for r in rows:
                r["kind"] = label
                nodes.append(r)
        # Final trim by PR — high-PR symbols win regardless of label.
        nodes.sort(key=lambda x: x.get("pagerank") or 0.0, reverse=True)
        nodes = nodes[:limit_nodes]
        node_ids = {n["id"] for n in nodes}

        edges: list[dict] = []
        # All edge types the UI's filter panel exposes. Without CONTAINS the
        # File→Function "click to expand" relationship has no edges to walk.
        # Filter at the DB level — pulling all edges then filtering in Python
        # ships millions of rows for nothing on a 200k-function codebase.
        ids_list = list(node_ids)
        for edge in self._ALL_GRAPH_EDGES:
            try:
                rows = self.db.fetch_all(
                    f"MATCH (a)-[r:{edge}]->(b) "
                    f"WHERE a.id IN $ids AND b.id IN $ids "
                    f"RETURN a.id AS src, b.id AS dst",
                    {"ids": ids_list},
                )
                for r in rows:
                    r["kind"] = edge
                    edges.append(r)
            except Exception:
                pass
        self._annotate_graph(nodes, edges)
        return {"nodes": nodes, "edges": edges}

    def files_dump(self, edges: bool = True) -> dict:
        """All File nodes + inter-file edges. Uncapped — used by the UI's
        Level-0 mode where the full file skeleton is intentional. Symbol
        nodes inside each file come in via lazy expansion on click.

        Edge set is restricted to file↔file relationships (IMPORTS, CO_CHANGED)
        because that's all that's structurally meaningful between Files —
        adding CALLS would cross node-kind boundaries and clutter the canvas.
        """
        try:
            rows = self.db.fetch_all(
                "MATCH (n:File) RETURN n.id AS id, n.path AS name, n.path AS file, "
                "coalesce(n.pagerank, 0.0) AS pagerank"
            )
        except Exception:
            rows = []
        nodes: list[dict] = []
        for r in rows:
            r["kind"] = "File"
            nodes.append(r)
        node_ids = {n["id"] for n in nodes}
        if not edges:
            return {"nodes": nodes, "edges": []}

        edges: list[dict] = []
        for edge in ("IMPORTS", "CO_CHANGED_WITH", "LINKS_TO"):
            try:
                erows = self.db.fetch_all(
                    f"MATCH (a:File)-[r:{edge}]->(b:File) RETURN a.id AS src, b.id AS dst"
                )
                for r in erows:
                    if r["src"] in node_ids and r["dst"] in node_ids:
                        r["kind"] = edge
                        edges.append(r)
            except Exception:
                pass
        self._annotate_graph(nodes, [])
        return {"nodes": nodes, "edges": edges}

    # =====================================================================
    # Analysis tools (context / detect_changes / repo_map / clusters /
    # routes / trace / health / history / rename). Cypher stays here; the
    # algorithmic parts live in insights.py / history.py.
    # =====================================================================

    def _annotate_graph(self, nodes: list[dict], edges: list[dict]) -> None:
        """Attach `cluster` (Community id) to nodes and `confidence` to CALLS
        edges of a graph payload, in place. No-op on pre-v3 DBs."""
        ids = [n["id"] for n in nodes if n.get("id") is not None]
        if ids and self._cap("community"):
            try:
                rows = self.db.fetch_all(
                    "MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id IN $ids "
                    "RETURN n.id AS id, c.id AS cid",
                    {"ids": ids},
                )
                m = {r["id"]: r["cid"] for r in rows}
                for n in nodes:
                    n["cluster"] = m.get(n["id"])
            except Exception:
                pass
        calls = [e for e in edges if e.get("kind") == "CALLS"]
        if calls and self._cap("calls_conf"):
            src = list({e["src"] for e in calls})
            conf: dict[tuple[int, int], float] = {}
            for r in self._call_edges(src, "out"):
                k = (r["src"], r["dst"])
                conf[k] = max(conf.get(k, 0.0), float(r["conf"] or 0.0))
            for e in calls:
                e["confidence"] = conf.get((e["src"], e["dst"]))

    # ---- clusters --------------------------------------------------------

    def list_clusters(self, limit: int = 100) -> dict:
        memo = getattr(self, "_clusters_memo", None)
        if memo is None:
            memo = self._clusters_memo = {}
        if int(limit) not in memo:
            memo[int(limit)] = self._list_clusters(int(limit))
        return memo[int(limit)]

    def _list_clusters(self, limit: int = 100) -> dict:
        import json as _json
        if not self._cap("community"):
            return {"clusters": [], "links": [], "reindex_required": True}
        rows = self.db.fetch_all(
            "MATCH (c:Community) RETURN c.id AS id, c.name AS name, c.size AS size, "
            "c.cohesion AS cohesion, c.top_members AS top, c.files AS files, "
            "c.pagerank AS pagerank ORDER BY c.size DESC, c.name LIMIT $lim",
            {"lim": int(limit)},
        )
        for r in rows:
            for k in ("top", "files"):
                try:
                    r[k] = _json.loads(r.get(k) or "[]")
                except Exception:
                    r[k] = []
            r["top_members"] = r.pop("top")
        links: list[dict] = []
        try:
            links = self.db.fetch_all(
                "MATCH (a:Function)-[:CALLS]->(b:Function), "
                "(a)-[:MEMBER_OF]->(ca:Community), (b)-[:MEMBER_OF]->(cb:Community) "
                "WHERE ca.id <> cb.id "
                "RETURN ca.id AS src, cb.id AS dst, count(*) AS count ORDER BY count DESC LIMIT 500"
            )
        except Exception:
            links = []
        return {"clusters": rows, "links": links}

    def cluster(self, id: int | None = None, name: str | None = None, limit: int = 200) -> dict:
        import json as _json
        if not self._cap("community"):
            return {"found": False, "reindex_required": True}
        c = None
        if id is not None:
            rows = self.db.fetch_all("MATCH (c:Community) WHERE c.id = $id RETURN c AS c", {"id": int(id)})
            c = rows[0]["c"] if rows else None
        if c is None and name:
            rows = self.db.fetch_all(
                "MATCH (c:Community) WHERE c.name = $n OR c.name CONTAINS $n RETURN c AS c, c.size AS s "
                "ORDER BY s DESC LIMIT 1", {"n": name})
            c = rows[0]["c"] if rows else None
        if c is None:
            return {"found": False}
        info = {k: v for k, v in c.items() if not k.startswith("_")}
        cid = int(info["id"])
        for k in ("top_members", "files"):
            try:
                info[k] = _json.loads(info.get(k) or "[]")
            except Exception:
                info[k] = []
        members = self.db.fetch_all(
            "MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE c.id = $id "
            "RETURN n.id AS id, label(n) AS kind, coalesce(n.name, n.path) AS name, "
            "n.qname AS qname, coalesce(n.file, n.path) AS file, n.line_start AS line, "
            "coalesce(n.pagerank, 0.0) AS pagerank ORDER BY pagerank DESC LIMIT $lim",
            {"id": cid, "lim": int(limit)},
        )
        api = self.db.fetch_all(
            "MATCH (a:Function)-[:CALLS]->(b:Function)-[:MEMBER_OF]->(c:Community) "
            "WHERE c.id = $id AND NOT EXISTS { MATCH (a)-[:MEMBER_OF]->(c) } "
            "RETURN b.name AS name, b.qname AS qname, b.file AS file, count(*) AS external_callers "
            "ORDER BY external_callers DESC LIMIT 25",
            {"id": cid},
        )
        deps = self.db.fetch_all(
            "MATCH (a:Function)-[:MEMBER_OF]->(c:Community), (a)-[:CALLS]->(b:Function)-[:MEMBER_OF]->(d:Community) "
            "WHERE c.id = $id AND d.id <> $id "
            "RETURN d.id AS id, d.name AS name, count(*) AS count ORDER BY count DESC LIMIT 20",
            {"id": cid},
        )
        users = self.db.fetch_all(
            "MATCH (a:Function)-[:MEMBER_OF]->(d:Community), (a)-[:CALLS]->(b:Function)-[:MEMBER_OF]->(c:Community) "
            "WHERE c.id = $id AND d.id <> $id "
            "RETURN d.id AS id, d.name AS name, count(*) AS count ORDER BY count DESC LIMIT 20",
            {"id": cid},
        )
        info.update(found=True, members=members, api=api, depends_on=deps, used_by=users)
        return info

    # ---- context ---------------------------------------------------------

    def _pick_symbol(self, symbol: str, file: str | None = None) -> tuple[dict | None, list[dict]]:
        ids = self._symbol_ids(symbol, ("Function", "Class"), file)
        if not ids and file is None and "/" in (symbol or "") and "::" not in symbol:
            return None, []
        nodes = list(self._nodes(ids).values())
        nodes.sort(key=lambda n: (-(n.get("pagerank") or 0.0), n.get("is_test") or False, n["file"]))
        if not nodes:
            return None, []
        return nodes[0], nodes[1:]

    def _details(self, node: dict) -> dict:
        label = node["kind"]
        cols = "n.body AS body, n.llm_doc AS llm_doc"
        if label == "Function":
            cols += ", n.signature AS signature"
        if self._cap("history"):
            cols += (", n.first_seen_commit AS fc, n.first_seen_ts AS fts, "
                     "n.last_changed_commit AS lc, n.last_changed_ts AS lts")
        rows = self.db.fetch_all(f"MATCH (n:{label}) WHERE n.id = $id RETURN {cols}", {"id": node["id"]})
        return rows[0] if rows else {}

    def _handlers_among(self, ids: list[int]) -> list[dict]:
        out: list[dict] = []
        if not ids:
            return out
        for label, cap in (("Route", "routes"), ("Tool", "tools")):
            if not self._cap(cap):
                continue
            try:
                for r in self.db.fetch_all(
                    f"MATCH (r:{label})-[:HANDLES]->(f:Function) WHERE f.id IN $ids "
                    f"RETURN r.id AS rid, r.name AS name, f.id AS fid, f.name AS handler, r.file AS file",
                    {"ids": ids},
                ):
                    r["kind"] = label.lower()
                    out.append(r)
            except Exception:
                pass
        return out

    def _no_inbound(self, ids: list[int]) -> set[int]:
        if not ids:
            return set()
        rows = self.db.fetch_all(
            "MATCH (f:Function) WHERE f.id IN $ids AND NOT EXISTS { MATCH ()-[:CALLS]->(f) } "
            "AND NOT coalesce(f.is_test, false) RETURN f.id AS id",
            {"ids": ids},
        )
        return {r["id"] for r in rows}

    def _tests_for(self, ids: list[int], depth: int = 3, min_conf: float = 0.0) -> list[dict]:
        out: dict[tuple, dict] = {}
        if not ids:
            return []
        try:
            for r in self.db.fetch_all(
                "MATCH (t:Function)-[:TESTS]->(x) WHERE x.id IN $ids "
                "RETURN t.id AS id, t.name AS name, t.file AS file, t.line_start AS line",
                {"ids": ids},
            ):
                out[(r["name"], r["file"])] = dict(r, via="TESTS", depth=1)
        except Exception:
            pass
        dist, _p, _e = self._bfs_calls(ids, "in", depth, min_conf)
        for n in self._nodes([i for i, d in dist.items() if d > 0]).values():
            if n.get("is_test") and (n["name"], n["file"]) not in out:
                out[(n["name"], n["file"])] = {"id": n["id"], "name": n["name"], "file": n["file"],
                                               "line": n["line"], "via": "CALLS*", "depth": dist[n["id"]]}
        return sorted(out.values(), key=lambda r: (r["depth"], r["file"] or "", r["name"]))

    def context(self, symbol: str, file: str | None = None, tokens: int = 2000,
                min_confidence: float = 0.0) -> dict:
        """One-call 360 view of a symbol, trimmed to a token budget."""
        from docgraph.insights import estimate_tokens, signature_of, trim_sections
        from docgraph.summary import extract_docstring
        from docgraph.parse import detect_language
        from pathlib import Path as _P

        tokens = max(200, min(int(tokens or 2000), 32000))
        primary, alts = self._pick_symbol(symbol, file)
        if primary is None:
            sugg = []
            try:
                sugg = [{"name": r["name"], "file": r["file"], "kind": r["label"]}
                        for r in self.search(symbol, limit=5)]
            except Exception:
                pass
            return {"found": False, "symbol": symbol, "suggestions": sugg}
        pid = primary["id"]
        det = self._details(primary)
        body, _snip = self._redact(primary["file"], det.get("body") or "", None)
        lang = detect_language(_P(primary["file"] or "")) or ""
        doc = (extract_docstring(body or "", lang) or det.get("llm_doc") or "").strip()
        sig = signature_of(body or "", det.get("signature") or primary["name"])

        # callers (2 hops) ranked by PageRank x confidence
        seeds = [pid]
        if primary["kind"] == "Class":
            seeds += [r["id"] for r in self.db.fetch_all(
                "MATCH (c:Class)-[:CONTAINS]->(f:Function) WHERE c.id = $id RETURN f.id AS id", {"id": pid})]
        db = self.db

        # The sections below are independent: they run concurrently, each on
        # its own connection (and the git log in its own process).
        def conn(fn):
            def run():
                with db.thread_conn():
                    return fn()
            return run

        def do_callers():
            direct_in = {}
            for r in self._call_edges(seeds, "in", min_confidence):
                direct_in[r["src"]] = max(direct_in.get(r["src"], 0.0), float(r["conf"] or 0.0))
            bdist, _bp, _be = self._bfs_calls(seeds, "in", 2, min_confidence)
            cnodes = self._nodes([i for i, d in bdist.items() if d > 0])
            out = []
            for i, n in cnodes.items():
                conf = direct_in.get(i, 0.5)
                out.append(dict(n, depth=bdist[i], confidence=round(conf, 2),
                                score=(n["pagerank"] or 0.0) * conf / bdist[i]))
            out.sort(key=lambda n: (-n["score"], n["depth"], n["name"]))
            return out

        def do_callees():
            out = []
            out_rows = self._call_edges(seeds, "out", min_confidence)
            onodes = self._nodes({r["dst"] for r in out_rows})
            seen_c: set[int] = set()
            for r in sorted(out_rows, key=lambda r: (r.get("line") or 0)):
                n = onodes.get(r["dst"])
                if not n or n["id"] in seen_c or n["id"] in seeds:
                    continue
                seen_c.add(n["id"])
                out.append(dict(n, confidence=round(float(r["conf"] or 0.0), 2), method=r["method"]))
            return out

        def do_flows():
            # flows it participates in: ancestors that are route/tool handlers or entry points
            adist, _ap, _ae = self._bfs_calls(seeds, "in", 6, min_confidence)
            anc = list(adist)
            out = [dict(h, depth=adist.get(h["fid"], 0)) for h in self._handlers_among(anc)]
            entries = self._no_inbound([i for i in anc if i not in seeds])
            enodes = self._nodes(entries)
            for i in entries:
                n = enodes.get(i)
                if n:
                    out.append({"kind": "entry", "name": n["name"], "handler": n["name"],
                                "fid": i, "file": n["file"], "depth": adist[i]})
            out.sort(key=lambda f: (f["kind"] == "entry", f["depth"], f["name"]))
            return out

        def do_cluster():
            if not self._cap("community"):
                return None
            rows = db.fetch_all(
                "MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id = $id "
                "RETURN c.id AS id, c.name AS name, c.size AS size, c.cohesion AS cohesion",
                {"id": pid})
            return rows[0] if rows else None

        def do_similar():
            try:
                return db.fetch_all(
                    f"MATCH (a:{primary['kind']})-[r:SIMILAR_TO]-(b) WHERE a.id = $id "
                    f"RETURN DISTINCT b.name AS name, b.file AS file, b.line_start AS line, r.score AS score "
                    f"ORDER BY score DESC LIMIT 8", {"id": pid})
            except Exception:
                return []

        def do_commits():
            if self.cfg is None:
                return []
            try:
                from docgraph import history as _h
                own = self._owner_of(primary["file"])
                if own:
                    return _h.symbol_log(own[0], own[1], primary["line"] or 1,
                                         primary.get("line_end") or primary["line"] or 1, limit=5)
            except Exception:
                return []
            return []

        def do_rules():
            try:
                return [{"name": r.get("name") or r.get("path"), "description": r.get("description", "")}
                        for r in (self.rules_for(primary["file"]) or [])][:6]
            except Exception:
                return []
        jobs = {"callers": do_callers, "callees": do_callees, "tests": lambda: self._tests_for(seeds, 3, min_confidence),
                "flows": do_flows, "cluster": do_cluster, "similar": do_similar,
                "commits": do_commits, "rules": do_rules}
        futs = {k: _SEARCH_POOL2.submit(conn(fn)) for k, fn in jobs.items()}
        res_ = {k: f.result() for k, f in futs.items()}
        callers, callees, tests, flows = res_["callers"], res_["callees"], res_["tests"], res_["flows"]
        cluster, similar, commits, rules = res_["cluster"], res_["similar"], res_["commits"], res_["rules"]

        def loc(n: dict) -> str:
            return f"{n['file']}:{n.get('line') or '?'}"

        head = [f"`{primary['qname']}` ({primary['kind']}) at {loc(primary)}"
                f"-{primary.get('line_end') or ''}",
                f"    {sig}"]
        if doc:
            head.append(f"Doc: {doc[:600]}")
        if cluster:
            head.append(f"Cluster: {cluster['name']} (#{cluster['id']}, {cluster['size']} members)")
        if det.get("lc"):
            head.append(f"History: introduced {det.get('fc') or '?'}, last changed {det.get('lc')}")
        if alts:
            head.append("Other symbols with this name: " + ", ".join(loc(a) for a in alts[:5]))
        sections = [
            ("Definition", head),
            ("Callers", [f"- {c['name']} ({loc(c)}) depth {c['depth']} conf {c['confidence']}" for c in callers]),
            ("Callees", [f"- {c['name']} ({loc(c)}) conf {c['confidence']} [{c.get('method') or ''}]" for c in callees]),
            ("Tests", [f"- {t['name']} ({t['file']}:{t['line']}) via {t['via']}" for t in tests]),
            ("Flows", [f"- {f['kind']}: {f['name']} -> {f.get('handler')} (depth {f['depth']})" for f in flows]),
            ("Source", ["    " + ln for ln in (body or "").splitlines()[:80]]),
            ("Recent commits", [f"- {c['commit']} {c['date']} {c['author']}: {c['subject']}" for c in commits]),
            ("Similar", [f"- {s['name']} ({s['file']}:{s['line']}) {float(s['score'] or 0):.2f}" for s in similar]),
            ("Rules", [f"- {r['name']}: {(r['description'] or '')[:160]}" for r in rules]),
        ]
        text, keep, truncated = trim_sections(sections, tokens)

        def slim(n: dict, extra=()) -> dict:
            base = {k: n.get(k) for k in ("id", "name", "qname", "file", "line", "kind")}
            for k in extra:
                base[k] = n.get(k)
            return base

        return {
            "found": True,
            "symbol": dict(slim(primary, ("line_end", "pagerank")), signature=sig, doc=doc,
                           first_seen_commit=det.get("fc"), last_changed_commit=det.get("lc")),
            "alternatives": [slim(a) for a in alts[:10]],
            "callers": [slim(c, ("depth", "confidence")) for c in callers[:keep["Callers"]]],
            "callees": [slim(c, ("confidence", "method")) for c in callees[:keep["Callees"]]],
            "tests": tests[:keep["Tests"]],
            "flows": flows[:keep["Flows"]],
            "cluster": cluster,
            "commits": commits[:keep["Recent commits"]],
            "similar": similar[:keep["Similar"]],
            "rules": rules[:keep["Rules"]],
            "text": text,
            "tokens": estimate_tokens(text),
            "budget": tokens,
            "truncated": truncated,
        }

    def _owner_of(self, logical: str):
        if self.cfg is None or not logical:
            return None
        for root, prefix in self.cfg.roots_with_prefix():
            if prefix == "":
                return root, logical
            if logical.startswith(prefix):
                return root, logical[len(prefix):]
        return None

    # ---- detect_changes --------------------------------------------------

    def _mem_graph(self) -> dict:
        """The whole CALLS graph + handler / test / method maps in memory,
        loaded once per Retriever (rebuilt after every reindex). Used where
        a tool would otherwise issue hundreds of 1-hop queries. Adjacency
        is CSR numpy (Arrow in, ~16 bytes per edge), not a dict of tuples."""
        g = getattr(self, "_mem_graph_cache", None)
        if g is not None:
            return g
        with self._build_lock:
            g = getattr(self, "_mem_graph_cache", None)
            if g is not None:
                return g
            return self._mem_graph_build()

    def _mem_graph_build(self) -> dict:
        from collections import defaultdict as _dd
        src, dst, conf, _k = self.db.edge_endpoints(("CALLS",), with_conf=self._cap("calls_conf"),
                                                    from_label="Function", to_label="Function")
        if conf is None:
            conf = np.ones(len(src), dtype=np.float32)
        fwd = _CSRAdj(src, dst, conf)
        rev = _CSRAdj(dst, src, conf)
        handlers: dict[int, list] = _dd(list)
        for label, cap in (("Route", "routes"), ("Tool", "tools")):
            if not self._cap(cap):
                continue
            for r in self.db.fetch_all(
                    f"MATCH (r:{label})-[:HANDLES]->(f:Function) "
                    f"RETURN f.id AS fid, r.id AS rid, r.name AS name, r.file AS file, f.name AS handler"):
                handlers[r["fid"]].append({"kind": label.lower(), "rid": r["rid"], "name": r["name"],
                                           "file": r["file"], "handler": r["handler"]})
        tests_into: dict[int, list] = _dd(list)
        try:
            ts, tx, _c, _k2 = self.db.edge_endpoints(("TESTS",), from_label="Function")
            for t, x in zip(ts.tolist(), tx.tolist()):
                tests_into[x].append(t)
        except Exception:
            pass
        d = self.db.node_columns("Function", {"id": "n.id"}, where="n.is_test")
        is_test = set(d["id"].tolist()) if d.get("id") is not None else set()
        methods: dict[int, list] = _dd(list)
        cs, cf, _c, _k3 = self.db.edge_endpoints(("CONTAINS",), from_label="Class", to_label="Function")
        for c, f in zip(cs.tolist(), cf.tolist()):
            methods[c].append(f)
        g = {"fwd": fwd, "rev": rev, "handlers": handlers, "tests_into": tests_into,
             "is_test": is_test, "methods": methods}
        self._mem_graph_cache = g
        return g

    @staticmethod
    def _mem_bfs(seeds, adj: dict, depth: int, min_conf: float = 0.0) -> dict[int, int]:
        dist = {int(s): 0 for s in seeds}
        frontier = list(dist)
        for d in range(1, depth + 1):
            nxt = []
            for n in frontier:
                for m, c in adj.get(n, ()):
                    if c >= min_conf and m not in dist:
                        dist[m] = d
                        nxt.append(m)
            if not nxt:
                break
            frontier = nxt
        return dist

    def detect_changes(self, ref: str | None = None, diff: str | None = None, depth: int = 3,
                       min_confidence: float = 0.0, max_symbols: int = 200) -> dict:
        """Diff -> changed symbols -> callers, affected flows, tests to run,
        and an explainable risk score."""
        from docgraph.insights import overall_risk, overlaps, parse_unified_diff, symbol_risk

        depth = max(1, min(int(depth), 6))
        source = "diff" if diff else "git"
        files: list[dict] = []
        if diff:
            files = parse_unified_diff(diff)
        elif self.cfg is not None:
            from docgraph.git_tools import diff_text
            for prefix, text in diff_text(self.cfg, ref):
                for f in parse_unified_diff(text):
                    f["path"] = prefix + f["path"]
                    files.append(f)
        else:
            return {"ref": ref, "source": source, "error": "no diff and no repo config", "files": []}

        # PageRank percentile table (one Arrow column, cached per retriever
        # -- i.e. per index generation)
        prs = getattr(self, "_pr_sorted", None)
        if prs is None:
            col = self.db.fetch_arrow("MATCH (f:Function) RETURN coalesce(f.pagerank, 0.0) AS p").column("p")
            prs = np.sort(np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64))
            self._pr_sorted = prs

        def pct(p: float) -> float:
            if not len(prs):
                return 0.0
            return float(np.searchsorted(prs, float(p), "left")) / len(prs)

        changed: list[dict] = []
        for f in files:
            path = f["path"]
            f.setdefault("indexed", True)
            ents = []
            for label in ("Function", "Class"):
                test_col = "n.is_test" if label == "Function" else "false"
                ents += [dict(r, kind=label) for r in self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.file = $f RETURN n.id AS id, n.name AS name, "
                    f"n.qname AS qname, n.file AS file, n.line_start AS s, n.line_end AS e, "
                    f"coalesce(n.pagerank, 0.0) AS pagerank, {test_col} AS is_test",
                    {"f": path})]
            if not ents:
                f["indexed"] = bool(self.db.fetch_all(
                    "MATCH (x:File) WHERE x.path = $f RETURN x.id LIMIT 1", {"f": path}))
            for e in ents:
                if f["status"] == "deleted":
                    n_lines = (e["e"] or 0) - (e["s"] or 0) + 1
                else:
                    n_lines = overlaps(e["s"] or 0, e["e"] or 0, f["ranges"])
                if n_lines > 0:
                    changed.append(dict(e, changed_lines=n_lines, line=e["s"], file_status=f["status"]))
        # Prefer innermost symbols: drop a Class when one of its methods is listed
        method_files = {(c["file"], c["qname"].rsplit("::", 1)[0]) for c in changed if c["kind"] == "Function"}
        changed = [c for c in changed if not (c["kind"] == "Class" and (c["file"], c["qname"]) in method_files
                                              and c["changed_lines"] <= 2)]
        changed.sort(key=lambda c: -(c["pagerank"] or 0.0))
        changed = changed[:max_symbols]

        all_callers: dict[int, int] = {}
        flows: dict[tuple, dict] = {}
        tests: dict[tuple, dict] = {}
        mg = self._mem_graph()
        rev = mg["rev"]
        # First pass: pure in-memory walks per symbol; payloads fetched once.
        per_symbol: list[tuple[dict, list[int], dict, dict, list[dict], list[int], dict]] = []
        need: set[int] = set()
        for c in changed:
            seeds = [c["id"]]
            if c["kind"] == "Class":
                seeds += mg["methods"].get(c["id"], [])
            bdist = self._mem_bfs(seeds, rev, depth, min_confidence)
            callers = {i: d for i, d in bdist.items() if d > 0}
            for i, d in callers.items():
                all_callers[i] = min(d, all_callers.get(i, d))
            adist = self._mem_bfs(seeds, rev, 8, min_confidence)
            handlers = [dict(h, fid=fid) for fid in adist for h in mg["handlers"].get(fid, ())]
            entry_ids = [i for i in adist if i not in seeds and not rev.get(i)
                         and i not in mg["is_test"]]
            tdist = {i: d for i, d in adist.items() if d <= 4}
            tmap: dict[int, tuple[int, str]] = {}
            for s_ in seeds:
                for t in mg["tests_into"].get(s_, ()):
                    tmap.setdefault(t, (1, "TESTS"))
            for i, d in tdist.items():
                if d > 0 and i in mg["is_test"]:
                    tmap.setdefault(i, (d, "CALLS*"))
            need.update(entry_ids)
            need.update(tmap)
            per_symbol.append((c, seeds, callers, adist, handlers, entry_ids, tmap))
        payload = self._nodes(need)
        for c, seeds, callers, adist, handlers, entry_ids, tmap in per_symbol:
            for h in handlers:
                flows.setdefault((h["kind"], h["name"]), dict(h, depth=adist.get(h["fid"], 0),
                                                             via=[]))["via"].append(c["name"])
            for i in entry_ids:
                n = payload.get(i)
                if n:
                    flows.setdefault(("entry", n["qname"]), {"kind": "entry", "name": n["name"],
                                                             "handler": n["name"], "fid": i,
                                                             "file": n["file"], "depth": adist[i],
                                                             "via": []})["via"].append(c["name"])
            for tid, (d, via) in tmap.items():
                n = payload.get(tid)
                if not n:
                    continue
                tests.setdefault((n["name"], n["file"]), {
                    "id": tid, "name": n["name"], "file": n["file"], "line": n["line"],
                    "via": via, "depth": d, "covers": []})["covers"].append(c["name"])
            c["risk"] = symbol_risk(
                pagerank_pct=pct(float(c["pagerank"] or 0.0)),
                n_callers=len(callers),
                n_routes=len(handlers),
                n_entries=len(entry_ids),
                n_tests=len(tmap),
                changed_lines=int(c["changed_lines"]),
                is_test=bool(c.get("is_test")),
            )
            c["callers"] = len(callers)
            c["tests"] = len(tmap)
        # Tests living in changed files run too
        for f in files:
            for r in self.db.fetch_all(
                "MATCH (t:Function) WHERE t.file = $f AND t.is_test "
                "RETURN t.id AS id, t.name AS name, t.file AS file, t.line_start AS line",
                {"f": f["path"]},
            ):
                tests.setdefault((r["name"], r["file"]), dict(r, via="changed_file", depth=0, covers=[]))
        cnodes = self._nodes(list(all_callers))
        callers_out = sorted(
            [dict({k: n.get(k) for k in ("id", "name", "qname", "file", "line", "pagerank")},
                  depth=all_callers[i]) for i, n in cnodes.items()],
            key=lambda r: (r["depth"], -(r["pagerank"] or 0.0)))
        tests_out = sorted(tests.values(), key=lambda t: (t.get("depth", 0), t["file"] or "", t["name"]))
        risk = overall_risk([c["risk"] for c in changed])
        return {
            "ref": ref,
            "source": source,
            "files": [{k: f.get(k) for k in ("path", "status", "ranges", "added", "removed", "indexed", "diff")}
                      for f in files],
            "changed_symbols": [{k: c.get(k) for k in ("id", "name", "qname", "file", "line", "kind",
                                                        "changed_lines", "file_status", "callers",
                                                        "tests", "risk", "pagerank")}
                                for c in changed],
            "callers": callers_out[:300],
            "flows": sorted(flows.values(), key=lambda f: (f["kind"] == "entry", f["depth"]))[:100],
            "tests": tests_out[:200],
            "test_command": self._test_command(tests_out),
            "risk": risk,
            "overlay": {"changed": [c["id"] for c in changed], "affected": list(all_callers)[:2000]},
        }

    @staticmethod
    def _test_command(tests: list[dict]) -> str:
        py = [t for t in tests if (t.get("file") or "").endswith(".py")]
        if not py:
            files = sorted({t["file"] for t in tests if t.get("file")})
            return " ".join(files)
        if len(py) <= 30:
            return "pytest " + " ".join(f"{t['file']}::{t['name']}" for t in py)
        return "pytest " + " ".join(sorted({t["file"] for t in py}))

    # ---- repo map --------------------------------------------------------

    REPO_MAP_POOL = 6000

    def repo_map(self, focus: list[str] | None = None, tokens: int = 1024,
                 exclude_tests: bool = True, include_focus_files: bool = True) -> dict:
        """Aider-style map: personalized PageRank biased to `focus` (files or
        symbols), signatures only, binary-searched to fit `tokens`."""
        from docgraph.insights import estimate_tokens, fit_to_budget, render_repo_map, signature_of

        tokens = max(64, min(int(tokens or 1024), 32000))
        focus = [f for f in (focus or []) if f]
        seeds: list[int] = []
        focus_files: set[str] = set()
        resolved: list[dict] = []
        for f in focus:
            if self.db.fetch_all("MATCH (x:File) WHERE x.path = $p RETURN x.id LIMIT 1", {"p": f}):
                focus_files.add(f)
                ids = [r["id"] for r in self.db.fetch_all(
                    "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.file = $p "
                    "RETURN n.id AS id", {"p": f})]
                seeds += ids
                resolved.append({"focus": f, "kind": "file", "ids": len(ids)})
            else:
                ids = self._symbol_ids(f, ("Function", "Class"))
                seeds += ids
                resolved.append({"focus": f, "kind": "symbol", "ids": len(ids)})
        ppr = self._maybe_ppr_ids(seeds) if seeds else None
        # Bounded candidate set: the top REPO_MAP_POOL by (personalized)
        # rank plus the focus files' symbols -- never every row of the graph.
        pool = self.REPO_MAP_POOL
        cols = ("n.id AS id, n.qname AS qname, n.file AS file, coalesce(n.pagerank, 0.0) AS pr, "
                "{t} AS is_test, '{k}' AS kind")
        rows: list[dict] = []
        if ppr:
            ids = np.asarray(getattr(ppr, "ids", []), dtype=np.int64)
            sc = np.asarray(getattr(ppr, "scores", []), dtype=np.float64)
            if len(ids) > pool:
                top = np.argpartition(-sc, pool - 1)[:pool]
                ids = ids[top]
            want = [int(i) for i in ids.tolist()]
            for label, t in (("Function", "n.is_test"), ("Class", "false")):
                for s_ in range(0, len(want), 5000):
                    rows += self.db.fetch_all(
                        f"MATCH (n:{label}) WHERE n.id IN $ids RETURN " + cols.format(t=t, k=label),
                        {"ids": want[s_:s_ + 5000]})
            if focus_files:
                have = {r["id"] for r in rows}
                for label, t in (("Function", "n.is_test"), ("Class", "false")):
                    rows += [r for r in self.db.fetch_all(
                        f"MATCH (n:{label}) WHERE n.file IN $f RETURN " + cols.format(t=t, k=label),
                        {"f": sorted(focus_files)}) if r["id"] not in have]
        else:
            for label, t in (("Function", "n.is_test"), ("Class", "false")):
                rows += self.db.fetch_all(
                    f"MATCH (n:{label}) RETURN " + cols.format(t=t, k=label)
                    + f" ORDER BY pr DESC LIMIT {pool}")
        scored = []
        for r in rows:
            if exclude_tests and (r.get("is_test") or "/test" in (r["file"] or "")):
                continue
            if not include_focus_files and r["file"] in focus_files:
                continue
            s = (ppr.get(r["id"], 0.0) if ppr else 0.0) or 0.0
            if not ppr:
                s = r["pr"] or 0.0
            elif r["file"] in focus_files:
                s += 1e-3  # focus files stay visible even without graph mass
            scored.append((s, r))
        scored.sort(key=lambda t: (-t[0], t[1]["qname"]))
        top = [r for _s, r in scored[:1500]]
        # fetch signatures for the candidates
        by_id = {r["id"]: r for r in top}
        for label in ("Function", "Class"):
            ids = [r["id"] for r in top if r["kind"] == label]
            if not ids:
                continue
            for d in self.db.fetch_all(
                f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id, n.name AS name, "
                f"n.line_start AS line, n.line_end AS line_end, substring(n.body, 1, 600) AS head",
                {"ids": ids},
            ):
                r = by_id[d["id"]]
                r["name"] = d["name"]
                r["line"] = d["line"]
                r["line_end"] = d["line_end"]
                r["signature"] = signature_of(d["head"] or "", d["name"] or "")
        class_qn = {r["qname"]: r for r in top if r["kind"] == "Class"}

        def render(chosen: list[dict]) -> str:
            entries = []
            have = {r["qname"] for r in chosen}
            for r in chosen:
                if "signature" not in r:
                    continue
                parts = (r["qname"] or "").split("::")
                parent = "::".join(parts[:-1]) if len(parts) >= 3 else None
                if parent and parent not in have and parent in class_qn and "signature" in class_qn[parent]:
                    p = class_qn[parent]
                    entries.append({"file": p["file"], "line": p["line"], "line_end": p["line"],
                                    "signature": p["signature"], "parent": None})
                    have.add(parent)
                entries.append({"file": r["file"], "line": r["line"], "line_end": r.get("line_end"),
                                "signature": r["signature"], "parent": parent})
            # de-dup class headers
            seen = set()
            uniq = []
            for e in entries:
                k = (e["file"], e["line"], e["signature"])
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(e)
            return render_repo_map(uniq)

        text, n = fit_to_budget(top, tokens, render)
        files = sorted({r["file"] for r in top[:n]})
        return {"text": text, "tokens": estimate_tokens(text), "budget": tokens,
                "symbols": n, "files": files, "focus": resolved,
                "ranking": "personalized" if seeds else "global"}

    def _maybe_ppr_ids(self, ids: list[int]) -> dict[int, float] | None:
        try:
            return self._ranker_().personalized(ids)
        except Exception:
            return None

    # ---- routes / tools --------------------------------------------------

    def route_map(self, filter: str | None = None, limit: int = 500) -> dict:
        out = {"routes": [], "tools": []}
        if not self._cap("routes"):
            out["reindex_required"] = True
            return out
        f = (filter or "").lower()
        for label, key in (("Route", "routes"), ("Tool", "tools")):
            cols = ("r.method AS method, r.path AS path" if label == "Route"
                    else "r.kind AS tool_kind, '' AS path")
            rows = self.db.fetch_all(
                f"MATCH (r:{label}) OPTIONAL MATCH (r)-[:HANDLES]->(h:Function) "
                f"RETURN r.id AS id, r.name AS name, {cols}, r.framework AS framework, "
                f"r.file AS file, r.line AS line, h.id AS handler_id, h.name AS handler, "
                f"h.qname AS handler_qname, h.file AS handler_file, h.line_start AS handler_line "
                f"ORDER BY r.file, r.line LIMIT $lim",
                {"lim": int(limit)},
            )
            if f:
                rows = [r for r in rows if f in (r["name"] or "").lower()
                        or f in (r.get("handler") or "").lower() or f in (r["file"] or "").lower()]
            for r in rows:
                if r.get("handler_id") is not None:
                    dist, _p, _e = self._bfs_calls([r["handler_id"]], "out", 2)
                    r["reach"] = len(dist) - 1
                else:
                    r["reach"] = 0
            out[key] = rows
        return out

    def api_impact(self, route: str, depth: int = 4, min_confidence: float = 0.0) -> dict:
        """What a route / MCP tool touches: handler, reachable functions,
        files, tests, and other routes sharing its dependencies."""
        depth = max(1, min(int(depth), 8))
        if not self._cap("routes"):
            return {"found": False, "reindex_required": True}
        r = None
        for label in ("Route", "Tool"):
            for q, p in ((f"MATCH (r:{label}) WHERE r.name = $n RETURN r AS r, '{label}' AS label", {"n": route}),
                         (f"MATCH (r:{label}) WHERE r.name CONTAINS $n OR r.path = $n RETURN r AS r, "
                          f"'{label}' AS label, r.line AS l ORDER BY l LIMIT 1", {"n": route})):
                try:
                    rows = self.db.fetch_all(q, p)
                except Exception:
                    rows = []
                if rows:
                    r = rows[0]
                    break
            if r:
                break
        if not r:
            return {"found": False, "route": route}
        label = r["label"]
        info = {k: v for k, v in r["r"].items() if not k.startswith("_")}
        hrows = self.db.fetch_all(
            f"MATCH (r:{label})-[:HANDLES]->(h:Function) WHERE r.id = $id RETURN h.id AS id",
            {"id": int(info["id"])})
        if not hrows:
            return {"found": True, "route": info, "kind": label.lower(), "handler": None,
                    "reachable": [], "files": [], "tests": [], "shared_with": []}
        hid = hrows[0]["id"]
        dist, parent, _e = self._bfs_calls([hid], "out", depth, min_confidence)
        nodes = self._nodes(list(dist))
        reach = sorted(
            [dict({k: n.get(k) for k in ("id", "name", "qname", "file", "line", "pagerank")},
                  depth=dist[i]) for i, n in nodes.items() if i != hid],
            key=lambda x: (x["depth"], -(x["pagerank"] or 0.0)))
        files: dict[str, int] = {}
        for x in reach:
            files[x["file"]] = files.get(x["file"], 0) + 1
        tests = self._tests_for([hid], 3, min_confidence)
        cov = self.db.fetch_all(
            "MATCH (t:Function)-[:TESTS]->(x:Function) WHERE x.id IN $ids "
            "RETURN DISTINCT t.name AS name, t.file AS file, t.line_start AS line, x.name AS covers",
            {"ids": list(dist)})
        direct = [x["id"] for x in reach if x["depth"] == 1]
        shared = []
        if direct:
            for lab in ("Route", "Tool"):
                try:
                    shared += self.db.fetch_all(
                        f"MATCH (o:{lab})-[:HANDLES]->(f:Function)-[:CALLS]->(g:Function) "
                        f"WHERE g.id IN $ids AND f.id <> $h "
                        f"RETURN o.name AS name, count(DISTINCT g) AS shared ORDER BY shared DESC LIMIT 15",
                        {"ids": direct, "h": hid})
                except Exception:
                    pass
        hn = nodes.get(hid, {})
        return {
            "found": True, "kind": label.lower(), "route": info,
            "handler": {k: hn.get(k) for k in ("id", "name", "qname", "file", "line")},
            "reachable": reach[:300],
            "files": sorted(({"file": f, "symbols": n} for f, n in files.items()),
                            key=lambda x: -x["symbols"]),
            "tests": tests + [dict(c, via="TESTS(reachable)") for c in cov
                              if (c["name"], c["file"]) not in {(t["name"], t["file"]) for t in tests}],
            "shared_with": shared,
        }

    # ---- trace -----------------------------------------------------------

    def trace(self, a: str, b: str, max_depth: int = 8, min_confidence: float = 0.0) -> dict:
        """Shortest directed path a -> b over CALLS plus member edges
        (Class -CONTAINS-> method, Function -INSTANTIATES-> Class). Python
        BFS, so no var-length Cypher is involved."""
        max_depth = max(1, min(int(max_depth), 15))
        A = self._symbol_ids(a, ("Function", "Class"))
        B = set(self._symbol_ids(b, ("Function", "Class")))
        out = {"from": a, "to": b, "found": False, "direction": None, "path": [], "edges": [],
               "hops": 0}
        if not A or not B:
            out["error"] = "unknown symbol: " + (a if not A else b)
            return out

        # Level-synchronous BFS on an in-memory CSR of CALLS + Class-CONTAINS->
        # Function + INSTANTIATES (built once per retriever, i.e. per index
        # generation): an unreachable target used to walk the whole graph with
        # three IN-list queries per hop (seconds); here a hop is a few numpy ops.
        tg = self._trace_graph()
        KIND = ("CALLS", "CONTAINS", "INSTANTIATES")

        def search(src: list[int], dst: set[int]):
            ids = tg["ids"]
            n = len(ids)
            parent: dict[int, tuple[int, str, float]] = {}
            if n == 0:
                return None, parent, len(src)
            def dense(xs) -> np.ndarray:
                q = np.asarray(sorted(set(xs)), dtype=np.int64)
                pos = np.searchsorted(ids, q)
                ok = pos < n
                pos, q = pos[ok], q[ok]
                return pos[ids[pos] == q]
            srcd = dense(src)
            dstm = np.zeros(n, dtype=bool)
            dstm[dense(dst)] = True
            seen = np.zeros(n, dtype=bool)
            seen[srcd] = True
            n_seen = len(set(src))
            par = np.full(n, -1, dtype=np.int64)
            pk = np.zeros(n, dtype=np.int8)
            pc = np.zeros(n, dtype=np.float32)
            frontier = np.unique(srcd)
            indptr, nbr, kind, conf = tg["indptr"], tg["nbr"], tg["kind"], tg["conf"]
            for _ in range(max_depth):
                if not len(frontier):
                    break
                starts = indptr[frontier]
                counts = indptr[frontier + 1] - starts
                total = int(counts.sum())
                if not total:
                    break
                owner = np.repeat(frontier, counts)
                offs = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
                e = offs + np.arange(total, dtype=np.int64)
                nb, kd, cf = nbr[e], kind[e], conf[e]
                keep = (cf >= min_confidence) | (kd == 1)
                owner, nb, kd, cf = owner[keep], nb[keep], kd[keep], cf[keep]
                fresh = ~seen[nb]
                owner, nb, kd, cf = owner[fresh], nb[fresh], kd[fresh], cf[fresh]
                if not len(nb):
                    break
                u, first = np.unique(nb, return_index=True)
                first.sort()
                nb, owner, kd, cf = nb[first], owner[first], kd[first], cf[first]
                seen[nb] = True
                n_seen += len(nb)
                par[nb], pk[nb], pc[nb] = owner, kd, cf
                hit = np.nonzero(dstm[nb])[0]
                if len(hit):
                    d = int(nb[hit[0]])
                    path = [d]
                    while par[path[-1]] >= 0:
                        path.append(int(par[path[-1]]))
                    path = path[::-1]
                    for x in path[1:]:
                        parent[int(ids[x])] = (int(ids[par[x]]), KIND[int(pk[x])], float(pc[x]))
                    return [int(ids[x]) for x in path], parent, n_seen
                frontier = nb
            return None, parent, n_seen

        path, parent, explored = search(A, B)
        direction = "forward"
        if path is None:
            path, parent, explored2 = search(list(B), set(A))
            explored += explored2
            direction = "reverse" if path else None
        out["explored"] = explored
        if not path:
            return out
        nodes = self._nodes(path)
        out.update(found=True, direction=direction, hops=len(path) - 1)
        out["path"] = [{k: nodes.get(i, {}).get(k) for k in ("id", "name", "qname", "file", "line", "kind")}
                       for i in path]
        out["edges"] = [{"src": parent[d][0], "dst": d, "kind": parent[d][1],
                         "confidence": round(parent[d][2], 3)} for d in path[1:]]
        return out

    def _trace_graph(self) -> dict:
        """CSR over CALLS (with confidence) + Class-CONTAINS->Function +
        Function-INSTANTIATES->Class, dense ids sorted; cached per retriever."""
        g = getattr(self, "_trace_graph_cache", None)
        if g is not None:
            return g
        with self._build_lock:
            if getattr(self, "_trace_graph_cache", None) is None:
                self._trace_graph_cache = self._trace_graph_build()
            return self._trace_graph_cache

    def _trace_graph_build(self) -> dict:
        parts = []
        has_conf = self._cap("calls_conf")
        a, b, c, _k = self.db.edge_endpoints(("CALLS",), with_conf=has_conf,
                                             from_label="Function", to_label="Function")
        parts.append((a, b, 0, c if c is not None else np.ones(len(a), np.float32)))
        a, b, _c, _k = self.db.edge_endpoints(("CONTAINS",), from_label="Class", to_label="Function")
        parts.append((a, b, 1, np.ones(len(a), np.float32)))
        inst_conf = "confidence" in self.db.table_props("INSTANTIATES") if self.db.has_table("INSTANTIATES") else False
        a, b, c, _k = self.db.edge_endpoints(("INSTANTIATES",), with_conf=inst_conf,
                                             from_label="Function", to_label="Class")
        parts.append((a, b, 2, c if c is not None else np.ones(len(a), np.float32)))
        src = np.concatenate([p[0] for p in parts]).astype(np.int64)
        dst = np.concatenate([p[1] for p in parts]).astype(np.int64)
        kind = np.concatenate([np.full(len(p[0]), p[2], np.int8) for p in parts])
        conf = np.concatenate([np.asarray(p[3], dtype=np.float32) for p in parts])
        ids = np.unique(np.concatenate([src, dst])) if len(src) else np.zeros(0, np.int64)
        s_d = np.searchsorted(ids, src)
        d_d = np.searchsorted(ids, dst)
        order = np.argsort(s_d, kind="stable")
        indptr = np.zeros(len(ids) + 1, dtype=np.int64)
        np.add.at(indptr, s_d + 1, 1)
        np.cumsum(indptr, out=indptr)
        return {"ids": ids, "indptr": indptr, "nbr": d_d[order], "kind": kind[order], "conf": conf[order]}

    # ---- health ----------------------------------------------------------

    # Above these sizes health() bounds its two super-linear parts: sampled
    # betweenness runs on the HEALTH_BRIDGE_NODES best-connected functions,
    # and the name-reference text scan (reads every file) is skipped.
    HEALTH_BRIDGE_NODES = 20_000
    HEALTH_TEXT_SCAN_FILES = 5_000

    def health(self, limit: int = 15, min_confidence: float = 0.5, refresh: bool = False) -> dict:
        """Hubs, bridges, dead code, import cycles, large functions and
        untested hotspots. Served from (in order) the per-retriever cache,
        `.docgraph/health.json` when it belongs to the current index
        generation, or -- right after a reindex -- the previous generation's
        report marked `refreshing: true` while the host recomputes it in the
        background. Computed from Arrow arrays otherwise."""
        key = (int(limit), float(min_confidence))
        if not refresh and self.workspace is not None and self.cfg is not None:
            try:
                self.workspace.note_health_use(self.cfg.repo_root, key)
            except Exception:
                pass
        cache = getattr(self, "_health_cache", {})
        if key in cache:
            return cache[key]
        if not refresh:
            stale = (getattr(self, "_health_stale", None) or {}).get(key)
            if stale is not None and self.workspace is not None and self.cfg is not None:
                try:
                    self.workspace.schedule_maintenance(self.cfg.repo_root)
                except Exception:
                    pass
                return dict(stale, refreshing=True)
            disk = self._health_disk(key)
            if disk is not None:
                cache[key] = disk
                self._health_cache = cache
                return disk
        result = self._health_compute(limit, min_confidence)
        cache[key] = result
        self._health_cache = cache
        self._health_disk(key, result)
        return result

    def _health_stamp(self):
        if self.cfg is None:
            return None
        try:
            import json as _json
            return _json.loads((self.cfg.data_dir / "state.json").read_text()).get("last_indexed_at")
        except Exception:
            return None

    def _health_disk(self, key: tuple, result: dict | None = None) -> dict | None:
        """Read (result=None) or write the persisted report for `key`."""
        if self.cfg is None:
            return None
        import json as _json
        path = self.cfg.data_dir / "health.json"
        k = f"{key[0]}:{key[1]}"
        stamp = self._health_stamp()
        if stamp is None:
            return None
        try:
            doc = _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            doc = {}
        if doc.get("stamp") != stamp:
            doc = {"stamp": stamp, "reports": {}}
        if result is None:
            return (doc.get("reports") or {}).get(k)
        doc.setdefault("reports", {})[k] = result
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(_json.dumps(doc), encoding="utf-8")
            import os as _os
            _os.replace(tmp, path)
        except Exception:
            pass
        return result

    def _health_compute(self, limit: int = 15, min_confidence: float = 0.5) -> dict:
        import re as _re
        from collections import Counter as _Counter

        fcols = self.db.node_columns("Function", {
            "id": "n.id", "name": "n.name", "qname": "n.qname", "file": "n.file",
            "line": "coalesce(n.line_start, 0)", "line_end": "coalesce(n.line_end, 0)",
            "pagerank": "coalesce(n.pagerank, 0.0)", "is_test": "coalesce(n.is_test, false)"})
        fid = fcols.get("id")
        if fid is None or not len(fid):
            fid = np.zeros(0, np.int64)
        fid = fid.astype(np.int64)
        n = len(fid)
        order = np.argsort(fid, kind="stable")
        sid = fid[order]
        names = fcols.get("name", np.zeros(0)).tolist()
        qnames = fcols.get("qname", np.zeros(0)).tolist()
        files = fcols.get("file", np.zeros(0)).tolist()
        lines = fcols.get("line", np.zeros(0)).astype(np.int64) if n else np.zeros(0, np.int64)
        line_end = fcols.get("line_end", np.zeros(0)).astype(np.int64) if n else np.zeros(0, np.int64)
        prs = fcols.get("pagerank", np.zeros(0)).astype(np.float64) if n else np.zeros(0)
        is_test = fcols.get("is_test", np.zeros(0)).astype(bool) if n else np.zeros(0, bool)

        def rows_of(ids: np.ndarray) -> np.ndarray:
            if not n or not len(ids):
                return np.full(len(ids), -1, dtype=np.int64)
            p = np.clip(np.searchsorted(sid, ids), 0, n - 1)
            return np.where(sid[p] == ids, order[p], -1)

        has_conf = self._cap("calls_conf")
        ca, cb, cc, _k = self.db.edge_endpoints(("CALLS",), with_conf=has_conf,
                                               from_label="Function", to_label="Function")
        if cc is None:
            cc = np.ones(len(ca), dtype=np.float32)
        ra, rb = rows_of(ca), rows_of(cb)
        keep = (ra >= 0) & (rb >= 0) & (cc >= min_confidence) & (ra != rb)
        ea, eb = ra[keep], rb[keep]
        if len(ea):     # DiGraph semantics: collapse duplicate edges
            k2 = np.unique(ea * n + eb)
            ea, eb = k2 // n, k2 % n
        in_deg = np.bincount(eb, minlength=n) if n else np.zeros(0, np.int64)
        out_deg = np.bincount(ea, minlength=n) if n else np.zeros(0, np.int64)
        inbound = np.zeros(n, dtype=bool)
        rbv = rb[rb >= 0]
        inbound[rbv] = True
        for rel, fl, tl, side in (("TESTS", None, "Function", "b"), ("HANDLES", None, "Function", "b"),
                                  ("IMPORTS_SYMBOL", "File", "Function", "b"),
                                  ("DECORATED_BY", None, "Function", "b"),
                                  ("CALLS_CANDIDATE", "Function", "Function", "b"),
                                  ("OVERRIDES", "Function", "Function", "a"),
                                  ("OVERRIDES", "Function", "Function", "b")):
            try:
                xa, xb, _c2, _k2 = self.db.edge_endpoints((rel,), from_label=fl, to_label=tl)
                r_ = rows_of(xa if side == "a" else xb)
                inbound[r_[r_ >= 0]] = True
            except Exception:
                pass

        def row(i: int, **extra) -> dict:
            return dict({"id": int(fid[i]), "name": names[i], "qname": qnames[i], "file": files[i],
                         "line": int(lines[i]), "pagerank": float(prs[i])}, **extra)

        prod = np.nonzero(~is_test)[0] if n else np.zeros(0, np.int64)
        # hubs
        deg = in_deg + out_deg if n else np.zeros(0, np.int64)
        hubs = prod[np.argsort(-deg[prod], kind="stable")][:limit] if len(prod) else prod
        hubs_out = [row(int(i), fan_in=int(in_deg[i]), fan_out=int(out_deg[i])) for i in hubs if deg[i] > 0]
        # bridges (sampled betweenness on the undirected production call graph,
        # restricted to the best-connected functions on very big graphs)
        bridges_out = []
        prod_mask = np.zeros(n, dtype=bool)
        prod_mask[prod] = True
        um = prod_mask[ea] & prod_mask[eb] if len(ea) else np.zeros(0, bool)
        ua, ub = ea[um], eb[um]
        active = np.unique(np.concatenate([ua, ub])) if len(ua) else np.zeros(0, np.int64)
        bounded = len(active) > self.HEALTH_BRIDGE_NODES
        if bounded:
            udeg = np.bincount(np.concatenate([ua, ub]), minlength=n)
            best = active[np.argsort(-udeg[active], kind="stable")][:self.HEALTH_BRIDGE_NODES]
            sel = np.zeros(n, dtype=bool)
            sel[best] = True
            m2 = sel[ua] & sel[ub]
            ua, ub = ua[m2], ub[m2]
        # same sampling / normalisation as networkx.betweenness_centrality,
        # on arrays (graphalgo.py): node order = first appearance in the edges
        if len(ua):
            inter = np.empty(len(ua) * 2, dtype=np.int64)
            inter[0::2], inter[1::2] = ua, ub
            _u, first = np.unique(inter, return_index=True)
            nodes_ug = inter[np.sort(first)].tolist()
        else:
            nodes_ug = []
        if len(nodes_ug) > 2:
            from docgraph.graphalgo import sampled_betweenness
            k = min(len(nodes_ug), 200 if len(nodes_ug) < 20000 else 32)
            bc = sampled_betweenness(nodes_ug, ua, ub, k, seed=7)
            for i, s in sorted(bc.items(), key=lambda t: -t[1])[:limit]:
                if s > 0:
                    bridges_out.append(row(int(i), betweenness=round(s, 4)))
        # text references (callbacks, module-level calls, registrations)
        tokens: _Counter = _Counter()
        texts: dict[str, list[str]] = {}
        file_rows = self.db.node_columns("File", {"p": "n.path"})
        all_files = file_rows["p"].tolist() if file_rows.get("p") is not None else []
        text_scan = self.cfg is not None and len(all_files) <= self.HEALTH_TEXT_SCAN_FILES
        if text_scan:
            for p in all_files:
                try:
                    full = self.cfg.path_for(p)
                    if full.stat().st_size > 1_500_000:
                        continue
                    t = full.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                texts[p] = t.splitlines()
                tokens.update(_re.findall(r"[A-Za-z_$][\w$]*", t))
        entry_names = {"main", "__init__", "setup", "teardown", "setUp", "tearDown", "run", "cli",
                       "handler", "lambda_handler", "app", "create_app", "make_app"}
        dead = []
        for i in prod.tolist():
            name = names[i] or ""
            if inbound[i] or name in entry_names or (name.startswith("__") and name.endswith("__")):
                continue
            parts = (qnames[i] or "").split("::")
            if len(parts) >= 3 and not name.startswith("_"):
                continue  # public methods: dynamic dispatch / framework hooks
            if texts:
                if tokens.get(name, 0) > 1:
                    continue  # referenced somewhere by name (callback, registration, export)
                flines = texts.get(files[i]) or []
                ln = int(lines[i] or 1)
                prev = flines[ln - 2].strip() if 2 <= ln <= len(flines) + 1 else ""
                if prev.startswith("@"):
                    continue  # registered by a decorator
                if ln - 1 < len(flines) and flines[ln - 1].lstrip().startswith("export"):
                    continue
            dead.append(row(i, reason="no inbound calls/references and name not referenced elsewhere"
                            if texts else "no inbound calls/references"))
        dead.sort(key=lambda r: (r["file"] or "", r["line"] or 0))
        # import cycles: shortest cycles through each strongly connected
        # component (scipy SCC + bounded BFS, graphalgo.short_cycles)
        from docgraph.graphalgo import short_cycles
        ia, ib = [], []
        t_imp = self.db.fetch_arrow("MATCH (a:File)-[:IMPORTS]->(b:File) RETURN a.path AS a, b.path AS b")
        for x, y in zip(t_imp.column("a").to_pylist(), t_imp.column("b").to_pylist()):
            if x != y:
                ia.append(x)
                ib.append(y)
        cycles = short_cycles(ia, ib, max(limit, 25), max_len=6)
        # large functions
        span = line_end - lines if n else np.zeros(0, np.int64)
        sizes = prod[np.argsort(-span[prod], kind="stable")] if len(prod) else prod
        large = [row(int(i), lines=int(span[i]) + 1) for i in sizes[:limit] if span[i] + 1 >= 60]
        # untested hotspots: top-PageRank production functions no test reaches (3 hops)
        tested = np.zeros(n, dtype=bool)
        try:
            ta, tb, _c3, _k3 = self.db.edge_endpoints(("TESTS",), from_label="Function", to_label="Function")
            r_ = rows_of(tb)
            tested[r_[r_ >= 0]] = True
        except Exception:
            pass
        fwd = _CSRAdj(ea, eb, np.ones(len(ea), np.float32)) if len(ea) else None
        reached = is_test.copy()
        frontier = np.nonzero(is_test)[0]
        for _ in range(3):
            if fwd is None or not len(frontier):
                break
            nxt = []
            for u in frontier.tolist():
                for v, _c in fwd.get(u, ()):
                    if not reached[v]:
                        reached[v] = True
                        nxt.append(v)
            frontier = np.asarray(nxt, dtype=np.int64)
        tested |= reached
        ranked = prod[np.argsort(-prs[prod], kind="stable")] if len(prod) else prod
        top_n = ranked[:max(limit * 4, len(ranked) // 10)]
        untested = [row(int(i), fan_in=int(in_deg[i])) for i in top_n if not tested[i]][:limit]
        result = {
            "summary": {"functions": n, "production": int(len(prod)), "tests": int(is_test.sum()),
                        "call_edges": int(len(ea)), "dead_code": len(dead),
                        "import_cycles": len(cycles), "untested_hotspots": len(untested),
                        "min_confidence": min_confidence,
                        "bridges_sampled_on": int(min(len(active), self.HEALTH_BRIDGE_NODES)),
                        "name_reference_scan": bool(text_scan)},
            "hubs": hubs_out,
            "bridges": bridges_out,
            "dead_code": dead[:max(limit * 3, 30)],
            "import_cycles": cycles[:limit],
            "large_functions": large,
            "untested_hotspots": untested,
        }
        return result

    # ---- history ---------------------------------------------------------

    def symbol_history(self, name: str, file: str | None = None, limit: int = 10) -> dict:
        import datetime as _dt
        import json as _json
        from docgraph import history as _h

        def day(ts) -> str | None:
            try:
                return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).date().isoformat() if ts else None
            except Exception:
                return None

        ids = self._symbol_ids(name, ("Function", "Class"), file)
        nodes = sorted(self._nodes(ids).values(), key=lambda n: -(n["pagerank"] or 0.0))
        out = []
        for n in nodes[:5]:
            det = self._details(n) if self._cap("history") else {}
            rec = {k: n.get(k) for k in ("id", "name", "qname", "file", "line", "line_end", "kind")}
            rec.update(first_seen_commit=det.get("fc") or None, first_seen=day(det.get("fts")),
                       last_changed_commit=det.get("lc") or None, last_changed=day(det.get("lts")))
            own = self._owner_of(n["file"])
            rec["log"] = []
            if own and len(out) < 3:
                try:
                    rec["log"] = _h.symbol_log(own[0], own[1], n["line"] or 1, n.get("line_end") or n["line"] or 1,
                                               limit=limit)
                except Exception:
                    rec["log"] = []
            out.append(rec)
        removed = []
        if self.cfg is not None:
            try:
                st = _json.loads((self.cfg.data_dir / "state.json").read_text())
                for r in st.get("removed_symbols") or []:
                    if r.get("name") == name or r.get("qname") == name:
                        removed.append(dict(r, removed=day(r.get("removed_at"))))
            except Exception:
                pass
        return {"name": name, "symbols": out, "removed": removed[-20:],
                "history_indexed": self._cap("history")}

    # ---- rename ----------------------------------------------------------

    def rename_plan(self, symbol: str, new_name: str, file: str | None = None,
                    include_text: bool = True, max_files: int = 20000) -> dict:
        """Edit plan (never writes): definition + graph-confirmed reference
        lines (source "graph"), plus word-boundary text matches elsewhere
        (source "text", lower confidence)."""
        import re as _re

        plan = {"symbol": symbol, "new_name": new_name, "dry_run": True, "edits": [],
                "definitions": [], "warnings": [], "counts": {"graph": 0, "text": 0}, "files": []}
        if not _re.fullmatch(r"[A-Za-z_$][\w$]*", new_name or ""):
            plan["error"] = "new_name is not a valid identifier"
            return plan
        base = symbol.split("::")[-1]
        ids = self._symbol_ids(symbol, ("Function", "Class"), file)
        defs = list(self._nodes(ids).values())
        if not defs:
            plan["error"] = f"no symbol named {symbol!r}"
            return plan
        if len(defs) > 1 and not file:
            plan["warnings"].append(
                f"{len(defs)} symbols are named {base!r}; pass file= to scope the rename. "
                "Text matches may belong to any of them.")
        clash = self.db.fetch_all(
            "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.name = $n "
            "AND n.file IN $files RETURN n.qname AS q LIMIT 5",
            {"n": new_name, "files": list({d["file"] for d in defs})})
        if clash:
            plan["warnings"].append(f"{new_name!r} already exists: " + ", ".join(r["q"] for r in clash))
        graph_lines: set[tuple[str, int]] = {(d["file"], int(d["line"] or 0)) for d in defs}
        fn_ids = [d["id"] for d in defs if d["kind"] == "Function"]
        cl_ids = [d["id"] for d in defs if d["kind"] == "Class"]
        related: set[str] = {d["file"] for d in defs}
        if fn_ids:
            for r in self.db.fetch_all(
                "MATCH (a:Function)-[r:CALLS]->(b:Function) WHERE b.id IN $ids RETURN a.file AS f, r.line AS l",
                {"ids": fn_ids}):
                graph_lines.add((r["f"], int(r["l"] or 0)))
                related.add(r["f"])
        if cl_ids:
            for q in ("MATCH (a:Function)-[r:INSTANTIATES]->(b:Class) WHERE b.id IN $ids RETURN a.file AS f, r.line AS l",):
                for r in self.db.fetch_all(q, {"ids": cl_ids}):
                    graph_lines.add((r["f"], int(r["l"] or 0)))
                    related.add(r["f"])
        for r in self.db.fetch_all(
            "MATCH (f:File)-[:IMPORTS_SYMBOL]->(x) WHERE x.id IN $ids RETURN f.path AS f",
            {"ids": ids}):
            related.add(r["f"])
        plan["definitions"] = [{k: d.get(k) for k in ("name", "qname", "file", "line", "kind")} for d in defs]
        if self.cfg is None:
            plan["warnings"].append("no repo config: cannot read files")
            return plan
        pat = _re.compile(r"(?<![\w$])" + _re.escape(base) + r"(?![\w$])")
        files = sorted(related) if not include_text else [
            r["p"] for r in self.db.fetch_all("MATCH (f:File) RETURN f.path AS p ORDER BY p")][:max_files]
        for p in files:
            if self.cfg.ai_blocked_logical(p):
                continue
            try:
                full = self.cfg.path_for(p)
                if full.stat().st_size > 1_500_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if base not in text:
                continue
            for n, ln in enumerate(text.splitlines(), start=1):
                m = pat.search(ln)
                if not m:
                    continue
                src = "graph" if (p, n) in graph_lines else "text"
                conf = 1.0 if src == "graph" else (0.6 if p in related else 0.3)
                if src == "text" and not include_text:
                    continue
                plan["edits"].append({
                    "file": p, "line": n, "column": m.start() + 1,
                    "before": ln, "after": pat.sub(new_name, ln),
                    "source": src, "confidence": conf,
                })
                plan["counts"][src] += 1
        plan["files"] = sorted({e["file"] for e in plan["edits"]})
        missing = [f"{f}:{l}" for f, l in graph_lines
                   if not any(e["file"] == f and e["line"] == l for e in plan["edits"])]
        if missing:
            plan["warnings"].append("graph references whose line no longer contains the name "
                                    "(index stale?): " + ", ".join(sorted(missing)[:10]))
        return plan

    # ---- index info ------------------------------------------------------

    def index_info(self) -> dict:
        import json as _json
        from docgraph.db import SCHEMA_VERSION
        st: dict = {}
        if self.cfg is not None:
            try:
                st = _json.loads((self.cfg.data_dir / "state.json").read_text())
            except Exception:
                st = {}
        ver = st.get("schema_version", 1 if st else None)
        return {
            "schema_version": ver,
            "current_schema_version": SCHEMA_VERSION,
            "reindex_required": bool(st) and ver != SCHEMA_VERSION,
            "embedding_model": st.get("embedding_model"),
            "last_indexed_at": st.get("last_indexed_at"),
            "resolution": st.get("resolution") or {},
            "embed_cache": st.get("embed_cache") or {},
            "scan": st.get("scan") or {},
            "communities": st.get("communities"),
            "scip": st.get("scip") or {},
            "history_pending": len(st.get("history_pending") or []),
            "removed_symbols": len(st.get("removed_symbols") or []),
            "capabilities": {k: self._cap(k) for k in
                             ("calls_conf", "community", "routes", "tools", "cand", "history")},
        }


class _CSRAdj:
    """Read-only adjacency {node: [(neighbour, confidence), ...]} over CSR
    arrays; `.get(n, ())` builds only the requested row."""

    __slots__ = ("keys", "start", "nbr", "conf")

    def __init__(self, src: np.ndarray, dst: np.ndarray, conf: np.ndarray):
        order = np.argsort(src, kind="stable")
        s = np.asarray(src, dtype=np.int64)[order]
        self.nbr = np.asarray(dst, dtype=np.int64)[order]
        self.conf = np.asarray(conf, dtype=np.float32)[order]
        self.keys, self.start = np.unique(s, return_index=True)
        self.start = np.append(self.start, len(s)).astype(np.int64)

    def _row(self, n):
        i = int(np.searchsorted(self.keys, int(n)))
        if i < len(self.keys) and int(self.keys[i]) == int(n):
            return int(self.start[i]), int(self.start[i + 1])
        return None

    def get(self, n, default=()):
        r = self._row(n)
        if r is None:
            return default
        a, b = r
        return list(zip(self.nbr[a:b].tolist(), self.conf[a:b].tolist()))

    def __getitem__(self, n):
        v = self.get(n, None)
        if v is None:
            raise KeyError(n)
        return v

    def __contains__(self, n) -> bool:
        return self._row(n) is not None

    def degree(self, n) -> int:
        r = self._row(n)
        return 0 if r is None else r[1] - r[0]

    def __len__(self) -> int:
        return len(self.keys)

    def expand(self, frontier: np.ndarray):
        """(owner, neighbour, confidence) arrays of every row of `frontier`,
        rows in frontier order, each row in CSR order."""
        f = np.asarray(frontier, dtype=np.int64)
        if not len(f) or not len(self.keys):
            z = np.zeros(0, np.int64)
            return z, z, np.zeros(0, np.float32)
        i = np.searchsorted(self.keys, f)
        i[i >= len(self.keys)] = 0
        hit = self.keys[i] == f
        f, i = f[hit], i[hit]
        starts = self.start[i]
        counts = self.start[i + 1] - starts
        total = int(counts.sum())
        if not total:
            z = np.zeros(0, np.int64)
            return z, z, np.zeros(0, np.float32)
        owner = np.repeat(f, counts)
        base = np.repeat(starts - (np.cumsum(counts) - counts), counts)
        e = base + np.arange(total, dtype=np.int64)
        return owner, self.nbr[e], self.conf[e]
