"""Regression tests for the GPU→CPU embedder fallback.

DirectML on Windows can return ``DXGI_ERROR_DEVICE_HUNG`` (0x887A0006)
mid-inference when another process (e.g. llama.cpp) is hammering the
GPU. The session is poisoned after that failure; the embedder must
drop the GPU cache entry, reload on CPU, and retry once so the index
run still completes.
"""
from __future__ import annotations

import numpy as np
import pytest

from docgraph import embed as embed_mod
from docgraph.embed import Embedder, _MODEL_CACHE, _MODEL_CACHE_LOCK


class _FakeModel:
    """Stands in for fastembed.TextEmbedding.

    Returns deterministic 384-d zero vectors. If ``raise_first`` is set,
    the first call to ``embed`` raises an ORT-style Fail with the
    DirectML signature, then subsequent calls succeed (simulating the
    poisoned-GPU + reload-on-CPU dance).
    """

    def __init__(self, raise_first: bool = False) -> None:
        self.raise_first = raise_first
        self.calls = 0

    def embed(self, texts, batch_size=256, **_):
        self.calls += 1
        if self.raise_first and self.calls == 1:
            raise RuntimeError(
                "[ONNXRuntimeError] : 1 : FAIL : "
                "DmlExecutionProvider 887A0006 device hung"
            )
        for _ in texts:
            yield np.zeros(384, dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_cache():
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
    yield
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def test_falls_back_to_cpu_on_ort_failure(monkeypatch):
    gpu_model = _FakeModel(raise_first=True)
    cpu_model = _FakeModel(raise_first=False)

    # Whatever Embedder's `_ensure` returns first is treated as the GPU
    # session; after fallback it asks again, and we hand it the CPU one.
    state = {"gave_gpu": False}

    def _fake_ensure(self):
        if not state["gave_gpu"]:
            state["gave_gpu"] = True
            self._model = gpu_model
            return gpu_model
        self._model = cpu_model
        return cpu_model

    monkeypatch.setattr(Embedder, "_ensure", _fake_ensure)

    emb = Embedder("dummy-model", providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    out = emb.embed(["hello", "world"])

    assert out.shape == (2, 384)
    assert gpu_model.calls == 1, "GPU should have been tried exactly once"
    assert cpu_model.calls == 1, "CPU should have been used as fallback"
    assert emb.providers is None, "providers should be cleared after fallback"


def test_does_not_swallow_unrelated_errors(monkeypatch):
    """If the failure isn't ORT-flavored, propagate — don't silently drop
    the GPU and degrade. A model file corruption would re-fail on CPU
    and the caller deserves the original traceback."""
    bad = _FakeModel()

    def _raise(*_a, **_kw):
        raise ValueError("model file is corrupted")

    bad.embed = _raise  # type: ignore[assignment]
    monkeypatch.setattr(Embedder, "_ensure", lambda self: bad)

    emb = Embedder("dummy-model", providers=["DmlExecutionProvider"])
    with pytest.raises(ValueError, match="corrupted"):
        emb.embed(["x"])


def test_no_fallback_when_already_on_cpu(monkeypatch):
    """If the embedder was built without GPU providers, an ORT failure
    has no GPU to fall back from — propagate."""
    bad = _FakeModel(raise_first=True)
    monkeypatch.setattr(Embedder, "_ensure", lambda self: bad)

    emb = Embedder("dummy-model")  # no providers → CPU-only
    with pytest.raises(RuntimeError, match="ONNXRuntimeError"):
        emb.embed(["x"])


def test_cache_key_handles_provider_with_options_dict():
    """Regression: the DirectML adapter-id path passes
    ``("DmlExecutionProvider", {"device_id": 1})`` in the providers list.
    Earlier ``_cache_key`` did ``tuple(providers)``, which left the inner
    dict in the key — unhashable, so dict lookup raised TypeError before
    the embedder could even load. Lock in that the key is hashable."""
    from docgraph.embed import _cache_key, resolve_providers

    providers = resolve_providers(True, 1)
    assert any(isinstance(p, tuple) for p in providers), (
        "resolve_providers should wrap DmlExecutionProvider as a tuple"
    )
    key = _cache_key("BAAI/bge-small-en-v1.5", providers)
    # Must be hashable — store it as a dict key as a smoke test.
    {key: 1}
    # Same providers must produce the same key (cache must hit on reuse).
    assert _cache_key("BAAI/bge-small-en-v1.5", providers) == key
