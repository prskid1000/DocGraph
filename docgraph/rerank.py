"""Cross-encoder reranker (torch backend).

Bi-encoders (BGE-small) compute query/doc embeddings independently — fast
enough for thousands of candidates but blunt on token-level relevance.
A cross-encoder runs the (query, doc) pair through one model and captures
those interactions, which is much more accurate but only viable on a
top-K already-narrowed list.

Default model: Jina's tiny English reranker (~33 MB). Lazy-loaded on
first use so callers who never set `rerank=True` never pay the download.
"""
from __future__ import annotations

import gc
import logging
import time
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-tiny-en"
RERANK_TOP_K = 50  # how many candidates feed into the cross-encoder


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Reranker:
    """Lazy wrapper around `sentence_transformers.CrossEncoder`. Thread-safe.
    Public surface mirrors `Embedder`: `score()` / `rerank()` / `unload()`."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        torch_compile: bool = False,
    ) -> None:
        self.model_name = model_name or DEFAULT_RERANK_MODEL
        # device: None or "cpu" = CPU; "cuda" = NVIDIA GPU. Re-checked at
        # load time so a CPU-only box silently downgrades.
        self.device = device
        self.dtype = dtype
        # See `Embedder.torch_compile` — same opt-in trade-off.
        self.torch_compile = torch_compile
        self._lock = Lock()
        self._encoder: Any = None
        self._resolved_device: str = "cpu"
        # Monotonic timestamp of the last successful score call. Polled
        # by the workspace's idle unloader.
        self._last_used: float = 0.0

    def _ensure(self) -> Any:
        with self._lock:
            if self._encoder is not None:
                return self._encoder
            self._last_used = time.monotonic()
            import torch
            from sentence_transformers import CrossEncoder
            on_cuda = self.device == "cuda" and torch.cuda.is_available()
            if self.device == "cuda" and not on_cuda:
                log.warning(
                    "Reranker %s: cuda requested but unavailable — using CPU",
                    self.model_name,
                )
            self._resolved_device = "cuda" if on_cuda else "cpu"
            log.info(
                "Loading cross-encoder reranker %s on %s",
                self.model_name, self._resolved_device,
            )
            try:
                self._encoder = CrossEncoder(
                    self.model_name, device=self._resolved_device,
                )
            except Exception as exc:
                # Mirror Embedder's GPU-init fallback: if cuda load
                # fails, retry once on CPU before giving up.
                if self._resolved_device == "cuda":
                    log.warning(
                        "Reranker GPU init failed (%s); falling back to CPU",
                        exc,
                    )
                    self._resolved_device = "cpu"
                    self.device = None
                    self._encoder = CrossEncoder(self.model_name, device="cpu")
                else:
                    raise
            if self.torch_compile:
                # CrossEncoder wraps the inner HF model on `.model`; compile
                # that, not the wrapper.
                try:
                    if hasattr(self._encoder, "model") and self._encoder.model is not None:
                        self._encoder.model = torch.compile(
                            self._encoder.model, mode="reduce-overhead",
                        )
                        log.info("Reranker %s: torch.compile applied", self.model_name)
                except Exception as exc:
                    log.warning(
                        "Reranker %s: torch.compile failed (%s); "
                        "continuing without it", self.model_name, exc,
                    )
            return self._encoder

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one float per document. Higher = more relevant.

        Catches the same recoverable CUDA errors as `Embedder.embed()` and
        retries once on CPU — long-lived hosts shouldn't die because of
        a transient driver hiccup."""
        if not documents:
            return []
        daemon_scores = self._maybe_rerank_via_daemon(query, documents)
        if daemon_scores is not None:
            return daemon_scores
        try:
            return self._score_once(query, documents)
        except Exception as exc:
            if not self._is_recoverable_gpu_error(exc) or self._resolved_device != "cuda":
                raise
            log.warning(
                "GPU rerank failed (%s); dropping CUDA session and "
                "retrying on CPU.", exc,
            )
            self._fallback_to_cpu()
            return self._score_once(query, documents)

    def _maybe_rerank_via_daemon(
        self, query: str, documents: list[str],
    ) -> list[float] | None:
        """Route scoring to the shared daemon if daemon mode is on and the
        daemon's reranker matches this model. Returns scores, or None to
        signal in-process scoring. Never raises."""
        try:
            from docgraph import daemon as _daemon
            if not _daemon.client_enabled():
                return None
            spec_rerank = _daemon.client_rerank_model() or DEFAULT_RERANK_MODEL
            if spec_rerank != self.model_name:
                return None
            if _daemon.ensure_client_daemon() is None:
                return None
            scores = _daemon.rerank_via_daemon(query, documents)
            if scores is None or len(scores) != len(documents):
                return None
            self._last_used = time.monotonic()
            return scores
        except Exception:
            return None

    def _score_once(self, query: str, documents: list[str]) -> list[float]:
        import torch
        enc = self._ensure()
        self._last_used = time.monotonic()
        with torch.no_grad():
            scores = enc.predict(
                [(query, d) for d in documents],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        self._last_used = time.monotonic()
        return [float(s) for s in scores]

    @staticmethod
    def _is_recoverable_gpu_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in (
            "cuda out of memory", "out of memory",
            "cuda error", "cublas", "cudnn",
            "illegal memory access", "device-side assert",
            "no kernel image", "no cuda gpus",
        ))

    def _fallback_to_cpu(self) -> None:
        with self._lock:
            self._encoder = None
        self.device = None
        self._resolved_device = "cpu"
        _empty_cuda_cache()
        gc.collect()

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
        _empty_cuda_cache()
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
        cross-encoder. Mutates each item with `rerank_score` and updates
        `score_key`. Returns the list re-sorted by `rerank_score` desc."""
        k = top_k or RERANK_TOP_K
        head = items[:k]
        tail = items[k:]
        if not head:
            return items

        # Give the cross-encoder both name and snippet — cross-encoders
        # handle the unstructured join well, and 300 chars + name stays
        # inside the 512-token limit.
        docs = []
        for it in head:
            name = it.get("name") or ""
            text = it.get(text_key) or it.get("body") or ""
            docs.append(f"{name} {text}".strip())

        scores = self.score(query, docs)
        for it, s in zip(head, scores):
            s_val = float(s)
            it["rerank_score"] = s_val
            # Update the primary score key so the UI reflects the reranked order.
            it[score_key] = s_val

        head.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return head + tail
