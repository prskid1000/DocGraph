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
            rows = self.db.fetch_all(
                "MATCH (c:Chunk) RETURN c.parent_qname AS qname, c.embedding AS embedding"
            )
        except Exception:
            return {}
        if not rows:
            return {}
        mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
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
                f"n.embedding AS embedding, n.pagerank AS pagerank, n.llm_doc AS llm_doc"
            )
            if not rows:
                continue
            mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
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
                   max_nodes: int = 20000) -> tuple[dict[int, int], dict[int, int], list[dict]]:
        """BFS over CALLS. Returns (dist, parent, edges): dist[id] = hops
        from the nearest seed, parent[id] = predecessor on that path."""
        dist: dict[int, int] = {int(s): 0 for s in seeds}
        parent: dict[int, int] = {}
        edges: list[dict] = []
        frontier = set(dist)
        for d in range(1, max(0, depth) + 1):
            if not frontier:
                break
            nxt: set[int] = set()
            for r in self._call_edges(frontier, direction, min_conf):
                near, far = (r["src"], r["dst"]) if direction == "out" else (r["dst"], r["src"])
                edges.append(r)
                if far not in dist:
                    dist[far] = d
                    parent[far] = near
                    nxt.add(far)
            frontier = nxt
            if len(dist) > max_nodes:
                break
        return dist, parent, edges

    def _nodes(self, ids) -> dict[int, dict]:
        """Function/Class payloads by id."""
        ids = [int(i) for i in ids]
        out: dict[int, dict] = {}
        if not ids:
            return out
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
        fdist, _fp, _fe = self._bfs_calls(seeds, "out", depth, min_confidence)
        bdist, _bp, _be = self._bfs_calls(seeds, "in", depth, min_confidence)
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
        for r in self._call_edges(seeds, "out", min_confidence) + self._call_edges(seeds, "in", min_confidence):
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
        for edge in ("CALLS", "REFERENCES_", "SIMILAR_TO", "INHERITS", "TESTS"):
            try:
                rows = self.db.fetch_all(
                    f"MATCH (n)-[r:{edge}]-(other) WHERE n.name = $name AND other.name IS NOT NULL "
                    f"RETURN DISTINCT other.qname AS qname, other.name AS name, other.file AS file, "
                    f"other.line_start AS line, label(other) AS kind, "
                    f"coalesce(other.pagerank, 0.0) AS pagerank",
                    {"name": name},
                )
                for r in rows:
                    key = (r["qname"], 0)
                    if key in seen:
                        continue
                    seen.add(key)
                    r["via"] = edge
                    out.append(r)
            except Exception:
                pass
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
            for edge in EDGE_TYPES:
                try:
                    rows = self.db.fetch_all(
                        f"MATCH (a)-[r:{edge}]-(b) WHERE a.id IN $ids AND b.id IS NOT NULL "
                        f"RETURN a.id AS src, b.id AS dst",
                        {"ids": ids},
                    )
                except Exception:
                    continue
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
        for s in seeds:
            for r in self.db.fetch_all(
                "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.name = $s "
                "RETURN n.id AS id",
                {"s": s},
            ):
                seed_ids.append(r["id"])
        if not seed_ids:
            return {"nodes": [], "edges": []}

        # BFS — each level we expand via every requested edge type.
        seen: dict[int, int] = {sid: 0 for sid in seed_ids}  # id → min-distance
        frontier = set(seed_ids)
        edge_records: list[dict] = []
        for d in range(1, hops + 1):
            if not frontier:
                break
            next_frontier: set[int] = set()
            for edge in edges:
                conf_filter = ""
                params: dict = {"ids": list(frontier)}
                if (min_confidence > 0 and edge in ("CALLS", "INHERITS", "INSTANTIATES")
                        and "confidence" in self.db.table_props(edge)):
                    conf_filter = " AND coalesce(r.confidence, 1.0) >= $c"
                    params["c"] = float(min_confidence)
                try:
                    rows = self.db.fetch_all(
                        f"MATCH (a)-[r:{edge}]-(b) WHERE a.id IN $ids{conf_filter} "
                        f"RETURN a.id AS src, b.id AS dst",
                        params,
                    )
                except Exception:
                    continue
                for row in rows:
                    edge_records.append({"src": row["src"], "dst": row["dst"], "kind": edge})
                    if row["dst"] not in seen:
                        seen[row["dst"]] = d
                        next_frontier.add(row["dst"])
            frontier = next_frontier

        if not seen:
            return {"nodes": [], "edges": []}

        rows = self.db.fetch_all(
            "MATCH (n) WHERE n.id IN $ids AND (label(n) = 'Function' OR label(n) = 'Class') "
            "RETURN n.id AS id, n.name AS name, n.qname AS qname, n.file AS file, "
            "n.line_start AS line, label(n) AS kind, "
            "coalesce(n.pagerank, 0.0) AS pagerank",
            {"ids": list(seen.keys())},
        )
        nodes = []
        for r in rows:
            d = seen[r["id"]]
            # Higher score = closer + more central
            score = (1.0 / (d + 1)) + (r["pagerank"] or 0.0) * 0.5
            r["distance"] = d
            r["score"] = score
            nodes.append(r)
        nodes.sort(key=lambda x: x["score"], reverse=True)
        return {"nodes": nodes[:limit], "edges": edge_records}

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

    def _flow_for(self, entry: dict, max_chain_len: int, min_confidence: float = 0.0,
                  max_nodes: int = 50) -> dict | None:
        dist, parent, edges = self._bfs_calls([entry["id"]], "out", max_chain_len,
                                              min_confidence, max_nodes=max_nodes * 4)
        ids = [i for i, d in sorted(dist.items(), key=lambda t: (t[1], t[0])) if d > 0][:max_nodes]
        if not ids:
            return None
        nodes = self._nodes(ids + [entry["id"]])
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

    def processes(self, limit: int = 25, max_chain_len: int = 8,
                  min_confidence: float = 0.0) -> list[dict]:
        """Detect 'processes' = top-level execution flows. Entry points are
        route / MCP-tool handlers (framework maps) and Functions with no
        incoming CALLS, by PageRank. For each, walk forward through CALLS up
        to `max_chain_len` hops (BFS, confidence-filtered) and return the
        chain plus the call edges between its members (the UI draws them as
        a sequence diagram)."""
        max_chain_len = max(2, min(int(max_chain_len), 12))
        out: list[dict] = []
        for e in self._entry_points(limit):
            flow = self._flow_for(e, max_chain_len, min_confidence)
            if not flow:
                continue
            out.append(flow)
            if len(out) >= limit:
                break
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

    def files_dump(self) -> dict:
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
        direct_in = {}
        for r in self._call_edges(seeds, "in", min_confidence):
            direct_in[r["src"]] = max(direct_in.get(r["src"], 0.0), float(r["conf"] or 0.0))
        bdist, _bp, _be = self._bfs_calls(seeds, "in", 2, min_confidence)
        cnodes = self._nodes([i for i, d in bdist.items() if d > 0])
        callers = []
        for i, n in cnodes.items():
            conf = direct_in.get(i, 0.5)
            callers.append(dict(n, depth=bdist[i], confidence=round(conf, 2),
                                score=(n["pagerank"] or 0.0) * conf / bdist[i]))
        callers.sort(key=lambda n: (-n["score"], n["depth"], n["name"]))
        callees = []
        out_rows = self._call_edges(seeds, "out", min_confidence)
        onodes = self._nodes({r["dst"] for r in out_rows})
        seen_c: set[int] = set()
        for r in sorted(out_rows, key=lambda r: (r.get("line") or 0)):
            n = onodes.get(r["dst"])
            if not n or n["id"] in seen_c or n["id"] in seeds:
                continue
            seen_c.add(n["id"])
            callees.append(dict(n, confidence=round(float(r["conf"] or 0.0), 2), method=r["method"]))
        tests = self._tests_for(seeds, 3, min_confidence)
        # flows it participates in: ancestors that are route/tool handlers or entry points
        adist, _ap, _ae = self._bfs_calls(seeds, "in", 6, min_confidence)
        anc = list(adist)
        flows = [dict(h, depth=adist.get(h["fid"], 0)) for h in self._handlers_among(anc)]
        entries = self._no_inbound([i for i in anc if i not in seeds])
        enodes = self._nodes(entries)
        for i in entries:
            n = enodes.get(i)
            if n:
                flows.append({"kind": "entry", "name": n["name"], "handler": n["name"],
                              "fid": i, "file": n["file"], "depth": adist[i]})
        flows.sort(key=lambda f: (f["kind"] == "entry", f["depth"], f["name"]))
        cluster = None
        if self._cap("community"):
            rows = self.db.fetch_all(
                "MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id = $id "
                "RETURN c.id AS id, c.name AS name, c.size AS size, c.cohesion AS cohesion",
                {"id": pid})
            cluster = rows[0] if rows else None
        similar = []
        try:
            similar = self.db.fetch_all(
                f"MATCH (a:{primary['kind']})-[r:SIMILAR_TO]-(b) WHERE a.id = $id "
                f"RETURN DISTINCT b.name AS name, b.file AS file, b.line_start AS line, r.score AS score "
                f"ORDER BY score DESC LIMIT 8", {"id": pid})
        except Exception:
            similar = []
        commits: list[dict] = []
        if self.cfg is not None:
            try:
                from docgraph import history as _h
                own = self._owner_of(primary["file"])
                if own:
                    commits = _h.symbol_log(own[0], own[1], primary["line"] or 1,
                                            primary.get("line_end") or primary["line"] or 1, limit=5)
            except Exception:
                commits = []
        rules = []
        try:
            rules = [{"name": r.get("name") or r.get("path"), "description": r.get("description", "")}
                     for r in (self.rules_for(primary["file"]) or [])][:6]
        except Exception:
            rules = []

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
        a tool would otherwise issue hundreds of 1-hop queries."""
        g = getattr(self, "_mem_graph_cache", None)
        if g is not None:
            return g
        from collections import defaultdict as _dd
        conf = "coalesce(r.confidence, 1.0)" if self._cap("calls_conf") else "1.0"
        fwd: dict[int, list] = _dd(list)
        rev: dict[int, list] = _dd(list)
        for r in self.db.fetch_all(
                f"MATCH (a:Function)-[r:CALLS]->(b:Function) RETURN a.id AS a, b.id AS b, {conf} AS c"):
            c = float(r["c"] or 0.0)
            fwd[r["a"]].append((r["b"], c))
            rev[r["b"]].append((r["a"], c))
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
            for r in self.db.fetch_all("MATCH (t:Function)-[:TESTS]->(x) RETURN t.id AS t, x.id AS x"):
                tests_into[r["x"]].append(r["t"])
        except Exception:
            pass
        is_test = {r["id"] for r in self.db.fetch_all(
            "MATCH (f:Function) WHERE f.is_test RETURN f.id AS id")}
        methods: dict[int, list] = _dd(list)
        for r in self.db.fetch_all("MATCH (c:Class)-[:CONTAINS]->(f:Function) RETURN c.id AS c, f.id AS f"):
            methods[r["c"]].append(r["f"])
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
        import bisect
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

        # PageRank percentile table
        prs = sorted(float(r["p"] or 0.0) for r in self.db.fetch_all(
            "MATCH (f:Function) RETURN coalesce(f.pagerank, 0.0) AS p"))

        def pct(p: float) -> float:
            if not prs:
                return 0.0
            return bisect.bisect_left(prs, p) / len(prs)

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
        rows = self.db.fetch_all(
            "MATCH (n:Function) RETURN n.id AS id, n.qname AS qname, n.file AS file, "
            "coalesce(n.pagerank, 0.0) AS pr, n.is_test AS is_test, 'Function' AS kind"
        ) + self.db.fetch_all(
            "MATCH (n:Class) RETURN n.id AS id, n.qname AS qname, n.file AS file, "
            "coalesce(n.pagerank, 0.0) AS pr, false AS is_test, 'Class' AS kind"
        )
        ppr = self._maybe_ppr_ids(seeds) if seeds else None
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

        def expand(frontier: set[int]) -> list[tuple[int, int, str, float]]:
            res = [(r["src"], r["dst"], "CALLS", float(r["conf"] or 0.0))
                   for r in self._call_edges(frontier, "out", min_confidence)]
            ids = list(frontier)
            try:
                for r in self.db.fetch_all(
                    "MATCH (c:Class)-[:CONTAINS]->(f:Function) WHERE c.id IN $ids "
                    "RETURN c.id AS s, f.id AS d", {"ids": ids}):
                    res.append((r["s"], r["d"], "CONTAINS", 1.0))
                conf = ("coalesce(r.confidence, 1.0)" if "confidence" in self.db.table_props("INSTANTIATES")
                        else "1.0")
                for r in self.db.fetch_all(
                    f"MATCH (f:Function)-[r:INSTANTIATES]->(c:Class) WHERE f.id IN $ids "
                    f"RETURN f.id AS s, c.id AS d, {conf} AS c", {"ids": ids}):
                    if float(r["c"] or 0.0) >= min_confidence:
                        res.append((r["s"], r["d"], "INSTANTIATES", float(r["c"] or 0.0)))
            except Exception:
                pass
            return res

        def search(src: list[int], dst: set[int]):
            parent: dict[int, tuple[int, str, float]] = {}
            seen = set(src)
            frontier = set(src)
            for _ in range(max_depth):
                if not frontier:
                    break
                nxt = set()
                for s, d, kind, c in expand(frontier):
                    if d in seen:
                        continue
                    seen.add(d)
                    parent[d] = (s, kind, c)
                    if d in dst:
                        path = [d]
                        while path[-1] in parent:
                            path.append(parent[path[-1]][0])
                        return path[::-1], parent, len(seen)
                    nxt.add(d)
                frontier = nxt
            return None, parent, len(seen)

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

    # ---- health ----------------------------------------------------------

    def health(self, limit: int = 15, min_confidence: float = 0.5) -> dict:
        """Hubs, bridges, dead code, import cycles, large functions and
        untested hotspots. Computed on demand, cached per retriever (which
        is rebuilt after every reindex)."""
        import re as _re
        from collections import Counter as _Counter
        import networkx as nx

        key = (int(limit), float(min_confidence))
        cache = getattr(self, "_health_cache", {})
        if key in cache:
            return cache[key]
        funcs = {r["id"]: r for r in self.db.fetch_all(
            "MATCH (f:Function) RETURN f.id AS id, f.name AS name, f.qname AS qname, f.file AS file, "
            "f.line_start AS line, f.line_end AS line_end, coalesce(f.pagerank, 0.0) AS pagerank, "
            "coalesce(f.is_test, false) AS is_test")}
        has_conf = self._cap("calls_conf")
        conf = "coalesce(r.confidence, 1.0)" if has_conf else "1.0"
        calls = self.db.fetch_all(
            f"MATCH (a:Function)-[r:CALLS]->(b:Function) RETURN a.id AS a, b.id AS b, {conf} AS c")
        g = nx.DiGraph()
        g.add_nodes_from(funcs)
        for r in calls:
            if float(r["c"] or 0.0) >= min_confidence and r["a"] != r["b"]:
                g.add_edge(r["a"], r["b"])
        inbound_any: set[int] = {r["b"] for r in calls}
        for q in ("MATCH (t)-[:TESTS]->(x:Function) RETURN x.id AS id",
                  "MATCH (h)-[:HANDLES]->(x:Function) RETURN x.id AS id",
                  "MATCH (f:File)-[:IMPORTS_SYMBOL]->(x:Function) RETURN x.id AS id",
                  "MATCH (y)-[:DECORATED_BY]->(x:Function) RETURN x.id AS id",
                  "MATCH (y:Function)-[:CALLS_CANDIDATE]->(x:Function) RETURN x.id AS id",
                  "MATCH (x:Function)-[:OVERRIDES]->(y:Function) RETURN x.id AS id",
                  "MATCH (y:Function)-[:OVERRIDES]->(x:Function) RETURN x.id AS id"):
            try:
                inbound_any |= {r["id"] for r in self.db.fetch_all(q)}
            except Exception:
                pass

        def row(i: int, **extra) -> dict:
            f = funcs[i]
            return dict({k: f.get(k) for k in ("id", "name", "qname", "file", "line", "pagerank")}, **extra)

        prod = [i for i, f in funcs.items() if not f["is_test"]]
        # hubs
        hubs = sorted(prod, key=lambda i: -(g.in_degree(i) + g.out_degree(i)))[:limit]
        hubs_out = [row(i, fan_in=g.in_degree(i), fan_out=g.out_degree(i)) for i in hubs
                    if g.in_degree(i) + g.out_degree(i) > 0]
        # bridges (approximate betweenness on the undirected call graph)
        ug = g.subgraph(prod).to_undirected()
        ug.remove_nodes_from([n for n in list(ug.nodes) if ug.degree(n) == 0])
        bridges_out = []
        if ug.number_of_nodes() > 2:
            k = min(ug.number_of_nodes(), 200 if ug.number_of_nodes() < 20000 else 50)
            bc = nx.betweenness_centrality(ug, k=k, seed=7, normalized=True)
            for i, s in sorted(bc.items(), key=lambda t: -t[1])[:limit]:
                if s > 0:
                    bridges_out.append(row(i, betweenness=round(s, 4)))
        # text references (callbacks, module-level calls, registrations)
        tokens: _Counter = _Counter()
        texts: dict[str, list[str]] = {}
        if self.cfg is not None:
            for r in self.db.fetch_all("MATCH (f:File) RETURN f.path AS p"):
                p = r["p"]
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
        for i in prod:
            f = funcs[i]
            name = f["name"] or ""
            if i in inbound_any or name in entry_names or (name.startswith("__") and name.endswith("__")):
                continue
            parts = (f["qname"] or "").split("::")
            if len(parts) >= 3 and not name.startswith("_"):
                continue  # public methods: dynamic dispatch / framework hooks
            if texts:
                if tokens.get(name, 0) > 1:
                    continue  # referenced somewhere by name (callback, registration, export)
                lines = texts.get(f["file"]) or []
                ln = int(f["line"] or 1)
                prev = lines[ln - 2].strip() if 2 <= ln <= len(lines) + 1 else ""
                if prev.startswith("@"):
                    continue  # registered by a decorator
                if ln - 1 < len(lines) and lines[ln - 1].lstrip().startswith("export"):
                    continue
            dead.append(row(i, reason="no inbound calls/references and name not referenced elsewhere"
                            if texts else "no inbound calls/references"))
        dead.sort(key=lambda r: (r["file"] or "", r["line"] or 0))
        # import cycles
        ig = nx.DiGraph()
        for r in self.db.fetch_all("MATCH (a:File)-[:IMPORTS]->(b:File) RETURN a.path AS a, b.path AS b"):
            if r["a"] != r["b"]:
                ig.add_edge(r["a"], r["b"])
        cycles = []
        try:
            for cyc in nx.simple_cycles(ig, length_bound=6):
                cycles.append(cyc)
                if len(cycles) >= max(limit, 25):
                    break
        except TypeError:
            for comp in nx.strongly_connected_components(ig):
                if len(comp) > 1:
                    cycles.append(sorted(comp))
        cycles.sort(key=lambda c: (len(c), c))
        # large functions
        sizes = sorted(prod, key=lambda i: -((funcs[i]["line_end"] or 0) - (funcs[i]["line"] or 0)))
        large = [row(i, lines=(funcs[i]["line_end"] or 0) - (funcs[i]["line"] or 0) + 1)
                 for i in sizes[:limit] if (funcs[i]["line_end"] or 0) - (funcs[i]["line"] or 0) + 1 >= 60]
        # untested hotspots: top-PageRank production functions no test reaches
        tested: set[int] = set()
        try:
            tested |= {r["id"] for r in self.db.fetch_all(
                "MATCH (t:Function)-[:TESTS]->(x:Function) RETURN x.id AS id")}
        except Exception:
            pass
        test_ids = [i for i, f in funcs.items() if f["is_test"]]
        frontier = set(test_ids)
        reached = set(test_ids)
        for _ in range(3):
            nxt = set()
            for n in frontier:
                if n in g:
                    for m in g.successors(n):
                        if m not in reached:
                            reached.add(m)
                            nxt.add(m)
            frontier = nxt
        tested |= reached
        ranked = sorted(prod, key=lambda i: -(funcs[i]["pagerank"] or 0.0))
        top_n = ranked[:max(limit * 4, len(ranked) // 10)]
        untested = [row(i, fan_in=g.in_degree(i)) for i in top_n if i not in tested][:limit]
        result = {
            "summary": {"functions": len(funcs), "production": len(prod), "tests": len(test_ids),
                        "call_edges": g.number_of_edges(), "dead_code": len(dead),
                        "import_cycles": len(cycles), "untested_hotspots": len(untested),
                        "min_confidence": min_confidence},
            "hubs": hubs_out,
            "bridges": bridges_out,
            "dead_code": dead[:max(limit * 3, 30)],
            "import_cycles": cycles[:limit],
            "large_functions": large,
            "untested_hotspots": untested,
        }
        cache[key] = result
        self._health_cache = cache
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
