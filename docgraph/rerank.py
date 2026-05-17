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

import gc
import logging
import os
import time
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
        # Monotonic timestamp of the last successful score call. Polled
        # by the workspace's idle unloader.
        self._last_used: float = 0.0

    def _ensure(self):
        with self._lock:
            if self._encoder is None:
                self._last_used = time.monotonic()
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
        self._last_used = time.monotonic()
        scores = list(enc.rerank(query, documents))
        self._last_used = time.monotonic()
        return scores

    # ── Idle unload ─────────────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._encoder is not None

    def last_used(self) -> float:
        return self._last_used

    def unload(self) -> bool:
        """Drop the in-memory cross-encoder. Returns True if something
        was actually evicted. Idempotent."""
        with self._lock:
            if self._encoder is None:
                return False
            self._encoder = None
        gc.collect()
        log.info("Reranker %s: unloaded after idle", self.model_name)
        return True

    def rerank(
        self,
        query: str,
        items: list[dict],
        text_key: str = "snippet",
        score_key: str = "score",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank `items` by feeding `query` and `item[text_key]` to the
        cross-encoder. Mutates each item to add `rerank_score` and update
        `score_key`. Returns the list re-sorted by `rerank_score` desc."""
        k = top_k or RERANK_TOP_K
        head = items[:k]
        tail = items[k:]
        if not head:
            return items

        # For best results, give the cross-encoder both the name and the
        # snippet. Cross-encoders are good at handling this unstructured
        # join. 300 chars of snippet + name is well within the 512-token limit.
        docs = []
        for it in head:
            name = it.get("name") or ""
            text = it.get(text_key) or it.get("body") or ""
            docs.append(f"{name} {text}".strip())

        scores = self.score(query, docs)
        for it, s in zip(head, scores):
            s_val = float(s)
            it["rerank_score"] = s_val
            # Update the primary score key so the UI reflects the reranked order
            it[score_key] = s_val

        head.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return head + tail
