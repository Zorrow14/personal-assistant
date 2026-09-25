"""Local text embeddings with fastembed (ONNX on onnxruntime, no torch).

The only module that knows about fastembed. Everything runs on this machine;
the model is downloaded once and cached.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.core.interfaces import Embedder, RetrievalError
from jarvis.logging import get_logger

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

ModelFactory = Callable[[str, str | None], Any]

log = get_logger(__name__)


def _load_fastembed(model_name: str, cache_dir: str | None) -> Any:
    from fastembed import TextEmbedding  # heavy import; only when first needed

    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)


def _lookup_dim(model_name: str) -> int:
    from fastembed import TextEmbedding

    for info in TextEmbedding.list_supported_models():
        if info["model"] == model_name:
            return int(info["dim"])
    raise RetrievalError(
        f"unknown embedding model {model_name!r}; see fastembed's supported models "
        f"(default: {DEFAULT_MODEL})"
    )


class LocalEmbedder(Embedder):
    """`Embedder` backed by a fastembed model, loaded lazily once."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | None = None,
        model_factory: ModelFactory = _load_fastembed,
        dim_lookup: Callable[[str], int] = _lookup_dim,
    ) -> None:
        """
        Args:
            model: fastembed model name, e.g. "BAAI/bge-small-en-v1.5".
            cache_dir: Where the ONNX model is cached (fastembed defaults to the
                system temp folder, which may be wiped).
            model_factory: Builds the model (injectable for tests).
            dim_lookup: Returns the vector length for `model` (injectable for tests).
        """
        self.model_name = model
        self._cache_dir = cache_dir
        self._model_factory = model_factory
        self._dim_lookup = dim_lookup
        self._dim: int | None = None
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        """Vector length, known without loading the model."""
        if self._dim is None:
            self._dim = self._dim_lookup(self.model_name)
        return self._dim

    def load(self) -> None:
        """Load the model now (downloads it on first ever use).

        Raises:
            RetrievalError: If the model can't be downloaded or loaded.
        """
        with self._lock:
            if self._model is not None:
                return
            log.info("embedder.model_loading", model=self.model_name)
            started = time.perf_counter()
            cache = str(self._cache_dir) if self._cache_dir else None
            try:
                self._model = self._model_factory(self.model_name, cache)
            except Exception as exc:
                raise RetrievalError(
                    f"could not load embedding model {self.model_name!r}: {exc}"
                ) from exc
            log.info("embedder.model_loaded", seconds=round(time.perf_counter() - started, 2))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (passages) in one batch."""
        if not texts:
            return []
        self.load()
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the model's query instruction (e.g. bge's prefix)."""
        self.load()
        return next(iter(self._model.query_embed([text]))).tolist()


# TODO(phase-later): GeminiEmbedder(Embedder) as an opt-in cloud option. It must
# never be the default, because it would send note content off the machine.
