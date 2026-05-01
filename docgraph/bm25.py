"""Minimal BM25 + RRF for hybrid keyword search.

Built on demand from the existing graph — no reindex needed. The index is
constructed in memory on the first call to `Retriever.search()` with the
BM25 layer enabled, and cached on the Retriever instance.

Why custom and not `rank_bm25`:
  - rank_bm25's BM25Okapi.get_scores is O(N) per query term: it iterates
    every document for every term. That's ~1M ops/query at our scale.
    An inverted index is O(sum of posting-list lengths) which is much smaller.
  - Avoids a new dep. ~80 lines stays inline with the project's "minimal
    deps, stdlib first" stance.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

# Split on non-word boundaries AND on the gap before a capital letter so
# camelCase / PascalCase yield reasonable tokens (e.g., "GraphDB" -> "graph",
# "db"; "fetchAll" -> "fetch", "all"). Underscores and dots also split.
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for piece in _SPLIT_RE.split(text):
        if not piece:
            continue
        p = piece.lower()
        if len(p) >= 2:
            out.append(p)
    return out


class BM25Index:
    """In-memory inverted index. Build once per Retriever; query many times."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / max(1, self.N)) or 1.0
        # term -> list of (doc_idx, tf)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        # term -> document frequency
        df: dict[str, int] = defaultdict(int)
        for i, doc in enumerate(docs):
            tf = Counter(doc)
            for term, freq in tf.items():
                self.postings[term].append((i, freq))
                df[term] += 1
        # Precompute idf (BM25's variant; +1 to avoid negatives for very
        # common terms in small corpora).
        self.idf: dict[str, float] = {
            t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for t, n in df.items()
        }

    def score(self, query_tokens: Iterable[str]) -> list[float]:
        """Return one score per document (length self.N). Only iterates
        posting lists for query terms — much cheaper than scanning all docs."""
        scores = [0.0] * self.N
        seen: set[str] = set()
        for term in query_tokens:
            if term in seen:
                continue
            seen.add(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            k1, b, avgdl = self.k1, self.b, self.avgdl
            for i, tf in postings:
                dl = self.doc_len[i]
                num = tf * (k1 + 1)
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[i] += idf * num / denom
        return scores


def rrf_fuse(*ranked_lists: list[int], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion. Each input is an ordered list of doc IDs
    (best first). Returns {doc_id: fused_score}. The classic constant
    k=60 (Cormack et al. 2009) damps the contribution of low-ranked items
    enough that a strong signal in one ranker is rarely overruled by
    middling rank in another."""
    fused: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return dict(fused)
