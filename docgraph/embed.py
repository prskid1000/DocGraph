"""Embedding wrapper around fastembed (ONNX runtime, no torch dep)."""
from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, Iterator

import numpy as np
from fastembed import TextEmbedding

log = logging.getLogger(__name__)

# ONNX Runtime execution providers tried (in order) when GPU is requested.
# ORT picks the first one that's actually installed; CPU is the final
# fallback so an unsupported box still works. Adding `onnxruntime-gpu`
# (CUDA / TensorRT) or `onnxruntime-directml` (Windows GPU) lights up the
# corresponding entries — base `onnxruntime` only knows CPU.
GPU_PROVIDERS: list[str] = [
    "CUDAExecutionProvider",       # NVIDIA via onnxruntime-gpu
    "DmlExecutionProvider",        # Windows DirectML via onnxruntime-directml
    "CoreMLExecutionProvider",     # macOS via onnxruntime-silicon
    "ROCMExecutionProvider",       # AMD Linux
    "CPUExecutionProvider",        # always-available fallback
]

# Process-wide model cache. Keyed on (model_name, sorted providers tuple) so
# every Embedder() instance with the same config shares one loaded ONNX
# session. The model itself is ~100 MB resident; in multi-repo + watch + test
# scenarios we used to pay that for each Embedder. fastembed's session is
# thread-safe for the embed() call so sharing across asyncio handlers /
# the watch loop / Indexer is fine.
_MODEL_CACHE: dict[tuple, TextEmbedding] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _cache_key(model_name: str, providers: list[str] | None) -> tuple:
    return (model_name, tuple(providers) if providers else ())


def clear_model_cache() -> None:
    """Drop all cached embedding models. Tests may want this between
    suites that flip GPU providers; production code should never need it."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        providers: list[str] | None = None,
    ):
        self.model_name = model_name
        # When None, fastembed picks its default (CPU). When given, the list
        # is passed straight through to the underlying onnxruntime session.
        self.providers = providers
        self._model: TextEmbedding | None = None

    def _ensure(self) -> TextEmbedding:
        if self._model is not None:
            return self._model
        # Process-wide cache: a second Embedder() with matching config skips
        # the ~1s ONNX-session load and uses the existing in-memory model.
        # We resolve providers up front so the cache key reflects what was
        # actually selected, not what was requested (e.g. CUDA was requested
        # but only CPU is installed → both keys end up as CPU).
        providers = self._available_providers(self.providers) if self.providers else None
        key = _cache_key(self.model_name, providers)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._model = cached
                return cached
            if providers:
                log.info(
                    f"Loading embedding model {self.model_name} with providers={providers}..."
                )
                try:
                    model = TextEmbedding(
                        model_name=self.model_name, providers=providers
                    )
                    self._model = model
                    self._log_active_provider()
                except Exception as e:  # pragma: no cover - depends on installed ORT
                    log.warning(
                        f"GPU init failed ({e}); falling back to CPU. "
                        f"Install `onnxruntime-gpu` (NVIDIA) or `onnxruntime-directml` (Windows) for GPU."
                    )
                    model = TextEmbedding(model_name=self.model_name)
                    self._model = model
                    # Re-key under the CPU fallback so a second Embedder
                    # asking for GPU also lands on the same loaded session.
                    key = _cache_key(self.model_name, None)
            else:
                if self.providers:
                    log.warning(
                        f"GPU requested but no GPU provider is installed. "
                        f"Install `onnxruntime-gpu` (NVIDIA) or `onnxruntime-directml` (Windows). "
                        f"Falling back to CPU."
                    )
                else:
                    log.info(f"Loading embedding model {self.model_name}...")
                model = TextEmbedding(model_name=self.model_name)
                self._model = model
            _MODEL_CACHE[key] = self._model
        return self._model

    @staticmethod
    def _available_providers(requested: list[str]) -> list[str]:
        """Filter requested ORT providers against what's actually installed.
        ORT errors hard if you list an unavailable provider (e.g. CUDA without
        onnxruntime-gpu) instead of skipping it, so we pre-filter."""
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except Exception:
            return []
        kept = [p for p in requested if p in available]
        # Drop the all-CPU case so the caller takes the no-providers code path
        # (which lets fastembed apply its own default session options).
        if kept == ["CPUExecutionProvider"]:
            return []
        return kept

    def _log_active_provider(self) -> None:
        """Best-effort introspection of which ORT provider actually got selected.
        fastembed doesn't expose this directly so we dig through the model."""
        try:
            inner = getattr(self._model, "model", None) or getattr(self._model, "_model", None)
            sess = getattr(inner, "model", None) if inner is not None else None
            if sess is not None and hasattr(sess, "get_providers"):
                active = sess.get_providers()
                log.info(f"ONNX Runtime active providers: {active}")
        except Exception:
            pass

    def embed(
        self,
        texts: Iterable[str],
        batch_size: int = 256,
        on_progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Embed a list of texts. Returns float32 ndarray of shape (N, dim).
        Keeping vectors as numpy (~1.5 KB each) instead of Python list[float]
        (~12 KB each) is an 8x memory cut, which matters when embedding
        100k+ entities. If `on_progress` is given, it's called with the count
        of items completed each time fastembed yields a vector."""
        model = self._ensure()
        arrs: list[np.ndarray] = []
        for vec in model.embed(list(texts), batch_size=batch_size):
            arrs.append(np.asarray(vec, dtype=np.float32))
            if on_progress is not None:
                on_progress(1)
        if not arrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(arrs)

    def embed_iter(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[np.ndarray]:
        """Streaming variant: yield one float32 ndarray at a time."""
        model = self._ensure()
        for vec in model.embed(list(texts), batch_size=batch_size):
            yield np.asarray(vec, dtype=np.float32)

    @property
    def dim(self) -> int:
        # BGE-small = 384
        return 384
