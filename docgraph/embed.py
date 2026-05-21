"""Embedding wrapper around sentence-transformers (torch backend).

GPU is opt-in: `cfg.gpu` → `device="cuda"` if `torch.cuda.is_available()`,
else silent CPU fallback. The process-wide `_MODEL_CACHE` is keyed on
`(model_name, device, dtype)` so a second `Embedder(...)` with matching
config reuses one loaded model. CUDA OOM / illegal-memory / cuBLAS / cuDNN
errors mid-inference are caught by `embed()` — the cache entry is dropped,
the model reloads on CPU, retried once. See the
`Things that have broken before` entry in `CLAUDE.md` for the historical
DirectML failure mode that motivated the recovery wrapper.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable, Iterable, Iterator

import numpy as np

log = logging.getLogger(__name__)

# Process-wide model cache. Keyed on (model_name, device, dtype) so every
# `Embedder()` with the same config shares one loaded `SentenceTransformer`.
# A loaded BGE-small is ~130 MB resident; without the cache, multi-root +
# watch + tests would pay that for each construction.
_MODEL_CACHE: dict[tuple, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Common embedding dims so schema-init doesn't have to load the model just
# to size Kuzu's `embedding DOUBLE[N]` column. Unknown models fall through
# to a one-time `SentenceTransformer.get_sentence_embedding_dimension()`
# (cached back into this dict).
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5":                          384,
    "BAAI/bge-base-en-v1.5":                           768,
    "BAAI/bge-large-en-v1.5":                          1024,
    "BAAI/bge-small-zh-v1.5":                          512,
    "BAAI/bge-m3":                                     1024,
    "sentence-transformers/all-MiniLM-L6-v2":          384,
    "sentence-transformers/all-MiniLM-L12-v2":         384,
    "sentence-transformers/all-mpnet-base-v2":         768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2":   384,
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
    "intfloat/e5-small-v2":                            384,
    "intfloat/e5-base-v2":                             768,
    "intfloat/e5-large-v2":                            1024,
    "thenlper/gte-small":                              384,
    "thenlper/gte-base":                               768,
    "thenlper/gte-large":                              1024,
    "jinaai/jina-embeddings-v2-small-en":              512,
    "jinaai/jina-embeddings-v2-base-en":               768,
    "nomic-ai/nomic-embed-text-v1.5":                  768,
    "mixedbread-ai/mxbai-embed-large-v1":              1024,
}


def resolve_device(gpu: bool) -> str | None:
    """Return `"cuda"` if GPU is requested AND torch sees a usable CUDA
    device, else `None` (= CPU). Importing torch is deferred so a CPU-only
    install still works at module-import time."""
    if not gpu:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def _cache_key(model_name: str, device: str | None, dtype: str) -> tuple:
    return (model_name, device or "cpu", (dtype or "auto").lower())


def clear_model_cache() -> None:
    """Drop all cached embedding models. Tests that flip GPU between suites
    want this; production code never needs to call it."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
    _empty_cuda_cache()
    gc.collect()


def dim_for_model(model_name: str, default: int = 384) -> int:
    """Look up the output dim of an embedding model. Sizes Kuzu's
    `embedding DOUBLE[N]` column at schema-init time so on-disk array
    length matches the model. Falls back to `default` if the model
    isn't in the static table and can't be probed."""
    if not model_name:
        return default
    if model_name in _KNOWN_DIMS:
        return _KNOWN_DIMS[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, device="cpu")
        d = int(m.get_sentence_embedding_dimension() or default)
        _KNOWN_DIMS[model_name] = d
        del m
        gc.collect()
        return d
    except Exception:
        log.debug("dim_for_model lookup failed for %s; using default=%d",
                  model_name, default)
    return default


def _pick_dtype(pref: str, on_cuda: bool):
    """Translate a dtype preference (`"auto" | "fp16" | "bf16" | "fp32"`)
    to a torch dtype. Default: fp16 on CUDA (1.5–2× faster, negligible
    quality loss for cosine retrieval), fp32 on CPU (Intel/AMD CPUs don't
    accelerate fp16)."""
    import torch
    pref = (pref or "auto").lower()
    if pref in ("auto", "fp16"):
        return torch.float16 if on_cuda else torch.float32
    if pref == "bf16":
        return torch.bfloat16
    return torch.float32


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Embedder:
    """`sentence-transformers` wrapper with process-wide cache, idle-unload
    bookkeeping, daemon routing, and CPU fallback on CUDA failure.

    Public surface:
        embed(texts, batch_size=None, on_progress=None) -> np.ndarray
        embed_iter(texts, batch_size=None) -> Iterator[np.ndarray]
        unload() / is_loaded() / last_used() / dim
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        dtype: str = "auto",
        torch_compile: bool = False,
    ):
        self.model_name = model_name
        # device: None or "cpu" = CPU; "cuda" = NVIDIA GPU. We re-check
        # `torch.cuda.is_available()` at load time so a config asking for
        # CUDA on a CPU-only box silently downgrades instead of erroring.
        self.device = device
        self.dtype = dtype
        # torch.compile is opt-in: pays a one-time compile cost (~10-30s
        # on first invocation) for ~1.3-1.6× steady-state speedup on
        # GPU. Off by default so cold-start incrementals stay fast.
        self.torch_compile = torch_compile
        self._model: Any = None
        self._resolved_device: str = "cpu"
        # Monotonic timestamp of the last successful embed call. The
        # workspace's idle unloader polls this and evicts when stale.
        self._last_used: float = 0.0
        self._lock = threading.Lock()

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        # Mark a touch so a freshly-loaded session never sits at 0.0.
        self._last_used = time.monotonic()
        import torch
        on_cuda = self.device == "cuda" and torch.cuda.is_available()
        if self.device == "cuda" and not on_cuda:
            log.warning(
                "Embedder %s: cuda requested but unavailable — using CPU",
                self.model_name,
            )
        self._resolved_device = "cuda" if on_cuda else "cpu"
        torch_dtype = _pick_dtype(self.dtype, on_cuda)
        key = _cache_key(self.model_name, self._resolved_device, self.dtype)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._model = cached
                return cached
            log.info(
                "Loading embedding model %s on %s (%s)…",
                self.model_name, self._resolved_device, str(torch_dtype),
            )
            from sentence_transformers import SentenceTransformer
            # `model_kwargs={"torch_dtype": …}` is the supported way to
            # load straight into fp16 / bf16 — avoids a fp32 → fp16
            # round-trip.
            model = SentenceTransformer(
                self.model_name,
                device=self._resolved_device,
                model_kwargs={"torch_dtype": torch_dtype},
            )
            model.eval()
            if self.torch_compile:
                # `reduce-overhead` is the right mode for many small inference
                # calls (our hot path). `default` would be safer but slower.
                # Wrapping in try/except because compile fails on older
                # GPUs / Windows + Triton combos — fall back silently.
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    log.info("Embedder %s: torch.compile applied", self.model_name)
                except Exception as exc:
                    log.warning(
                        "Embedder %s: torch.compile failed (%s); "
                        "continuing without it", self.model_name, exc,
                    )
            self._model = model
            _MODEL_CACHE[key] = model
        return self._model

    def embed(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Embed a list of texts. Returns float32 ndarray of shape (N, dim).

        Daemon path: if `docgraph daemon start` is running on this host,
        route the call through it — the daemon holds a single warm session
        across CLI invocations. Daemon path skipped when `on_progress` is
        set since the daemon returns one batch and progress hooks expect
        per-vector ticks. Falls back transparently on any daemon failure."""
        texts_list = list(texts)
        if batch_size is None:
            batch_size = 64

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
            # torch CUDA can fail mid-inference: OOM, illegal memory
            # access, driver hang/crash. The session is poisoned afterwards;
            # rebuilding on the same device just re-fails. Drop the cache
            # entry, force CPU, retry once.
            if not self._is_recoverable_gpu_error(exc) or self.device != "cuda":
                raise
            log.warning(
                "GPU embed failed (%s); dropping CUDA session and "
                "retrying on CPU. %s", exc, _diagnose_gpu_error(exc),
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
        # Touch at start AND end so a long batch isn't evicted mid-flight.
        self._last_used = time.monotonic()
        if not texts_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch
        # `no_grad` keeps the autograd graph from holding tensors for
        # thousands of entities across a single index pass.
        if on_progress is None:
            with torch.no_grad():
                vecs = model.encode(
                    texts_list,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            self._last_used = time.monotonic()
            return np.asarray(vecs, dtype=np.float32)
        # Progress path: encode in our own batch loop so callers get
        # per-item ticks rather than one big tick at the end.
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts_list), batch_size):
                batch = texts_list[i:i + batch_size]
                vecs = model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                out.append(np.asarray(vecs, dtype=np.float32))
                for _ in range(len(batch)):
                    on_progress(1)
        self._last_used = time.monotonic()
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    @staticmethod
    def _is_recoverable_gpu_error(exc: BaseException) -> bool:
        """Heuristic: torch CUDA OOM, illegal-memory, cuBLAS / cuDNN /
        driver errors are recoverable by falling back to CPU. Other failures
        (model file missing, OOV tokenizer error, etc.) should propagate."""
        msg = str(exc).lower()
        return any(s in msg for s in (
            "cuda out of memory", "out of memory",
            "cuda error", "cublas", "cudnn",
            "illegal memory access", "device-side assert",
            "no kernel image", "no cuda gpus",
        ))

    def _fallback_to_cpu(self) -> None:
        """Drop CUDA session + cache entries; re-init on CPU on next embed."""
        with _MODEL_CACHE_LOCK:
            for key in (
                _cache_key(self.model_name, "cuda", self.dtype),
                _cache_key(self.model_name, self._resolved_device, self.dtype),
            ):
                _MODEL_CACHE.pop(key, None)
        self._model = None
        self.device = None
        self._resolved_device = "cpu"
        _empty_cuda_cache()
        gc.collect()

    def embed_iter(
        self, texts: Iterable[str], batch_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Streaming variant: yield one float32 ndarray per text."""
        vecs = self.embed(texts, batch_size=batch_size)
        for v in vecs:
            yield np.asarray(v, dtype=np.float32)

    # ── Idle unload ─────────────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._model is not None

    def last_used(self) -> float:
        return self._last_used

    def unload(self) -> bool:
        """Drop the in-memory model + cache entries. Returns True if
        something was actually evicted. Idempotent. Held under `self._lock`
        so a concurrent `_ensure()` either reloads cleanly or completes
        before we drop."""
        with self._lock:
            if self._model is None:
                return False
            with _MODEL_CACHE_LOCK:
                for key in (
                    _cache_key(self.model_name, self._resolved_device, self.dtype),
                    _cache_key(self.model_name, self.device or "cpu", self.dtype),
                ):
                    _MODEL_CACHE.pop(key, None)
            self._model = None
        # Native CUDA buffers stick around via pybind until we explicitly
        # ask torch to release them and run GC.
        _empty_cuda_cache()
        gc.collect()
        log.info("Embedder %s: unloaded after idle", self.model_name)
        return True

    @property
    def dim(self) -> int:
        """Output dim of the loaded model (BGE-small = 384, mpnet = 768, …)."""
        return dim_for_model(self.model_name, default=384)


def _diagnose_gpu_error(exc: BaseException) -> str:
    """One-line root-cause hint appended to the recovery warning. Useful
    for triage when a recovery fires in a long-running host log: was it
    OOM, a driver issue, or something else?"""
    msg = str(exc).lower()
    if "out of memory" in msg:
        return "Likely cause: CUDA out of memory (model + batch too big for VRAM)."
    if "illegal memory access" in msg or "device-side assert" in msg:
        return "Likely cause: CUDA illegal memory access — driver or kernel bug."
    if "cublas" in msg or "cudnn" in msg:
        return "Likely cause: cuBLAS / cuDNN error — version mismatch or driver bug."
    return "Cause unclear — see exception message above."
