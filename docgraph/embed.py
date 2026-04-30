"""Embedding wrapper around fastembed (ONNX runtime, no torch dep)."""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator

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
        if self._model is None:
            if self.providers:
                log.info(
                    f"Loading embedding model {self.model_name} with providers={self.providers}..."
                )
                try:
                    self._model = TextEmbedding(
                        model_name=self.model_name, providers=self.providers
                    )
                    self._log_active_provider()
                except Exception as e:  # pragma: no cover - depends on installed ORT
                    log.warning(
                        f"GPU init failed ({e}); falling back to CPU. "
                        f"Install `onnxruntime-gpu` (NVIDIA) or `onnxruntime-directml` (Windows) for GPU."
                    )
                    self._model = TextEmbedding(model_name=self.model_name)
            else:
                log.info(f"Loading embedding model {self.model_name}...")
                self._model = TextEmbedding(model_name=self.model_name)
        return self._model

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
    ) -> list[list[float]]:
        """Embed a list of texts. If `on_progress` is given, it's called with
        the count of items completed each time fastembed yields a vector —
        enables granular progress bars without forcing the caller to manage
        batches themselves."""
        model = self._ensure()
        out: list[list[float]] = []
        for vec in model.embed(list(texts), batch_size=batch_size):
            out.append(vec.tolist())
            if on_progress is not None:
                on_progress(1)
        return out

    def embed_iter(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[list[float]]:
        """Streaming variant: yield one vector at a time."""
        model = self._ensure()
        for vec in model.embed(list(texts), batch_size=batch_size):
            yield vec.tolist()

    @property
    def dim(self) -> int:
        # BGE-small = 384
        return 384
