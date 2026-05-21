"""Regression tests for the GPU→CPU embedder fallback.

torch CUDA can fail mid-inference: OOM, illegal memory access, cuBLAS /
cuDNN errors, driver crashes. The session is poisoned afterwards. The
embedder must drop the cached model, force CPU, and retry once so the
index run still completes.
"""
from __future__ import annotations

import numpy as np
import pytest

from docgraph.embed import Embedder, _MODEL_CACHE, _MODEL_CACHE_LOCK, _cache_key


class _FakeModel:
    """Stands in for `sentence_transformers.SentenceTransformer`.

    `encode()` returns deterministic zero vectors. If `raise_first` is
    set, the first call raises a torch-CUDA-style RuntimeError, then
    subsequent calls succeed (simulates poisoned-CUDA + reload-on-CPU)."""

    def __init__(self, raise_first: bool = False, dim: int = 384) -> None:
        self.raise_first = raise_first
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size=64, **_):
        self.calls += 1
        if self.raise_first and self.calls == 1:
            raise RuntimeError(
                "CUDA error: out of memory (CUDA out of memory)"
            )
        return np.zeros((len(texts), self.dim), dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_cache():
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
    yield
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def test_falls_back_to_cpu_on_cuda_failure(monkeypatch):
    gpu_model = _FakeModel(raise_first=True)
    cpu_model = _FakeModel(raise_first=False)

    state = {"gave_gpu": False}

    def _fake_ensure(self):
        if not state["gave_gpu"]:
            state["gave_gpu"] = True
            self._model = gpu_model
            return gpu_model
        self._model = cpu_model
        return cpu_model

    monkeypatch.setattr(Embedder, "_ensure", _fake_ensure)

    emb = Embedder("dummy-model", device="cuda")
    out = emb.embed(["hello", "world"])

    assert out.shape == (2, 384)
    assert gpu_model.calls == 1, "GPU should have been tried exactly once"
    assert cpu_model.calls == 1, "CPU should have been used as fallback"
    assert emb.device is None, "device should be cleared after fallback"


def test_does_not_swallow_unrelated_errors(monkeypatch):
    """If the failure doesn't look like a CUDA error, propagate — don't
    silently drop the GPU and degrade. A model file corruption would
    re-fail on CPU and the caller deserves the original traceback."""
    bad = _FakeModel()

    def _raise(*_a, **_kw):
        raise ValueError("model file is corrupted")

    bad.encode = _raise  # type: ignore[assignment]
    monkeypatch.setattr(Embedder, "_ensure", lambda self: bad)

    emb = Embedder("dummy-model", device="cuda")
    with pytest.raises(ValueError, match="corrupted"):
        emb.embed(["x"])


def test_no_fallback_when_already_on_cpu(monkeypatch):
    """If the embedder was built without `device="cuda"`, a CUDA-flavored
    failure has nowhere to fall back from — propagate."""
    bad = _FakeModel(raise_first=True)
    monkeypatch.setattr(Embedder, "_ensure", lambda self: bad)

    emb = Embedder("dummy-model")  # device=None → CPU-only
    with pytest.raises(RuntimeError, match="CUDA"):
        emb.embed(["x"])


def test_cache_key_is_hashable_across_dtype_variants():
    """`_cache_key` must produce hashable, equal-on-equal-inputs keys so
    multiple Embedder() instances with the same `(model, device, dtype)`
    triple share the underlying torch session."""
    k1 = _cache_key("BAAI/bge-small-en-v1.5", "cuda", "auto")
    k2 = _cache_key("BAAI/bge-small-en-v1.5", "cuda", "auto")
    k3 = _cache_key("BAAI/bge-small-en-v1.5", "cpu", "auto")
    k4 = _cache_key("BAAI/bge-small-en-v1.5", "cuda", "fp16")
    # Must be hashable — store as dict keys as a smoke test.
    {k1: 1, k3: 2, k4: 3}
    assert k1 == k2
    assert k1 != k3   # device differs
    assert k1 != k4   # dtype differs (after lowercasing in _cache_key)


def test_recoverable_error_classification_covers_cuda_signatures():
    """Locks the heuristic in `_is_recoverable_gpu_error` so a future
    refactor doesn't accidentally narrow what triggers CPU fallback."""
    for msg in (
        "CUDA out of memory: Tried to allocate 2.00 GiB",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "cuBLAS error: CUBLAS_STATUS_EXECUTION_FAILED",
        "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED",
        "device-side assert triggered",
    ):
        assert Embedder._is_recoverable_gpu_error(RuntimeError(msg)), msg

    for msg in (
        "model file not found",
        "tokenizer.json is malformed",
        "ValueError: too many positional arguments",
    ):
        assert not Embedder._is_recoverable_gpu_error(RuntimeError(msg)), msg
