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


def dim_for_model(model_name: str, default: int = 384) -> int:
    """Look up the output dim of a fastembed text-embedding model.

    Used to size Kuzu's `embedding DOUBLE[N]` columns at schema-init time
    so the on-disk array length matches whatever model the user picked.
    Falls back to `default` if the model isn't in fastembed's catalog
    (older fastembed, custom local model, mistyped name) — the indexer
    will then either succeed by chance (matching dim) or fail loudly on
    the first insert (mismatched dim). We prefer that to silent dim
    drift across a session."""
    if not model_name:
        return default
    try:
        for m in TextEmbedding.list_supported_models():
            if m.get("model") == model_name:
                d = m.get("dim")
                if isinstance(d, int) and d > 0:
                    return d
    except Exception:
        log.debug("dim_for_model lookup failed for %s; using default=%d", model_name, default)
    return default


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
        of items completed each time fastembed yields a vector.

        Daemon path: if `docgraph daemon start` is running on this host, route
        the embed call through it — the daemon holds a single warm ONNX session
        across all CLI invocations, cutting cold-start to a TCP round trip. We
        only use the daemon when no `on_progress` callback is set, since the
        daemon returns one batch and progress hooks expect per-vector ticks.
        Falls back transparently if the daemon is unreachable or replies
        malformed; never fails the caller's request.
        """
        texts_list = list(texts)
        if on_progress is None and texts_list:
            try:
                from docgraph import daemon as _daemon
                via = _daemon.embed_via_daemon(texts_list)
                if via is not None and via.shape[0] == len(texts_list):
                    return via.astype(np.float32, copy=False)
            except Exception:
                pass
        try:
            return self._run_embed(texts_list, batch_size, on_progress)
        except Exception as exc:
            # ORT GPU sessions can fail mid-inference for reasons unrelated
            # to docgraph: device hung (DXGI 0x887A0006 — common when
            # another process like llama.cpp is saturating the GPU),
            # OOM in shared VRAM, kernel cache eviction. The session is
            # poisoned after such a failure; rebuilding it on the same
            # provider just re-fails. Drop the cache entry, force CPU,
            # retry once.
            if not self._is_recoverable_ort_error(exc) or not self.providers:
                raise
            log.warning(
                "GPU embed failed (%s); dropping DirectML session and "
                "retrying on CPU. Likely cause: another GPU process "
                "(LLM) is contending for the device.", exc,
            )
            self._fallback_to_cpu()
            return self._run_embed(texts_list, batch_size, on_progress)

    def _run_embed(
        self,
        texts_list: list[str],
        batch_size: int,
        on_progress: Callable[[int], None] | None,
    ) -> np.ndarray:
        model = self._ensure()
        arrs: list[np.ndarray] = []
        for vec in model.embed(texts_list, batch_size=batch_size):
            arrs.append(np.asarray(vec, dtype=np.float32))
            if on_progress is not None:
                on_progress(1)
        if not arrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(arrs)

    @staticmethod
    def _is_recoverable_ort_error(exc: BaseException) -> bool:
        """Heuristic: ORT's pybind11_state errors and DXGI device-hung
        codes are the ones we want to recover from. Other failures (e.g.
        a malformed model file) won't get better on CPU and should
        propagate."""
        try:
            import onnxruntime as ort  # noqa: F401
            from onnxruntime.capi.onnxruntime_pybind11_state import Fail as ORTFail
            if isinstance(exc, ORTFail):
                return True
        except Exception:
            pass
        msg = str(exc)
        return ("ONNXRuntimeError" in msg
                or "DmlExecutionProvider" in msg
                or "887A0006" in msg)

    def _fallback_to_cpu(self) -> None:
        """Drop GPU session + cache entry; re-init on CPU on next embed."""
        with _MODEL_CACHE_LOCK:
            old_providers = (
                self._available_providers(self.providers) if self.providers else None
            )
            for key in (_cache_key(self.model_name, old_providers),
                        _cache_key(self.model_name, self.providers)):
                _MODEL_CACHE.pop(key, None)
        self._model = None
        self.providers = None

    def embed_iter(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[np.ndarray]:
        """Streaming variant: yield one float32 ndarray at a time."""
        model = self._ensure()
        for vec in model.embed(list(texts), batch_size=batch_size):
            yield np.asarray(vec, dtype=np.float32)

    @property
    def dim(self) -> int:
        # BGE-small = 384
        return 384
