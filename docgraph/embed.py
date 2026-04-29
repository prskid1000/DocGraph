"""Embedding wrapper around fastembed (ONNX runtime, no torch dep)."""
from __future__ import annotations

import logging
from typing import Iterable

from fastembed import TextEmbedding

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    def _ensure(self) -> TextEmbedding:
        if self._model is None:
            log.info(f"Loading embedding model {self.model_name}...")
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: Iterable[str], batch_size: int = 256) -> list[list[float]]:
        model = self._ensure()
        # fastembed returns a generator of np.ndarray
        out: list[list[float]] = []
        for vec in model.embed(list(texts), batch_size=batch_size):
            out.append(vec.tolist())
        return out

    @property
    def dim(self) -> int:
        # BGE-small = 384
        return 384
