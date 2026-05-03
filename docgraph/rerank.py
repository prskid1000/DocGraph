"""Cross-encoder reranker.

Bi-encoders (BGE-small) compute query/doc embeddings independently, which is
fast enough to score thousands of candidates but loses precision on subtle
relevance signals. A cross-encoder runs the (query, doc) pair through one
model, capturing token-level interactions — much more accurate but only
viable on a top-K already-narrowed candidate list.

We default to Jina's tiny English reranker (~33 MB ONNX). Lazy-loaded on
first use so users who never set rerank=True never pay the download.
"""
from __future__ import annotations

import logging
import os
from threading import Lock

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-tiny-en"
RERANK_TOP_K = 50  # how many candidates to feed into the cross-encoder


class Reranker:
    """Lazy wrapper around fastembed's TextCrossEncoder. Thread-safe."""

    def __init__(
        self,
        model_name: str | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_name = model_name or DEFAULT_RERANK_MODEL
        self.providers = providers
        self._lock = Lock()
        self._encoder = None

    def _ensure(self):
        with self._lock:
            if self._encoder is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                log.info(
                    f"Loading cross-encoder reranker: {self.model_name}"
                    + (f" (providers={self.providers})" if self.providers else "")
                )
                try:
                    self._encoder = TextCrossEncoder(
                        model_name=self.model_name,
                        providers=self.providers if self.providers else None,
                    )
                except Exception as exc:
                    if self.providers:
                        log.warning(
                            f"Reranker GPU init failed ({exc!s}); falling back to CPU"
                        )
                        self.providers = None
                        self._encoder = TextCrossEncoder(model_name=self.model_name)
                    else:
                        raise
            return self._encoder

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one float per document. Higher = more relevant."""
        if not documents:
            return []
        enc = self._ensure()
        return list(enc.rerank(query, documents))

    def rerank(
        self,
        query: str,
        items: list[dict],
        text_key: str = "snippet",
        score_key: str = "score",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank `items` by feeding `query` and `item[text_key]` to the
        cross-encoder. Mutates each item to add `rerank_score`. Returns the
        list re-sorted by `rerank_score` desc."""
        k = top_k or RERANK_TOP_K
        head = items[:k]
        tail = items[k:]
        if not head:
            return items
        docs = [str(it.get(text_key) or it.get("body") or it.get("name") or "") for it in head]
        scores = self.score(query, docs)
        for it, s in zip(head, scores):
            it["rerank_score"] = float(s)
        head.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return head + tail
