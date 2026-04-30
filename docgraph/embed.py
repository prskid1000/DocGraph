"""Embedding wrapper around fastembed (ONNX runtime, no torch dep)."""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator

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
