"""LocalEmbedder (fake model), real ChromaStore on a tmp dir, and factory selection.

Chroma here is the embedded local database; we pass vectors in, so no model is
downloaded and nothing touches the network.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.config import Settings
from jarvis.core.interfaces import RetrievalError
from jarvis.memory.embedder import LocalEmbedder
from jarvis.memory.factory import build_memory, create_embedder, create_vector_store
from jarvis.memory.vector_store import ChromaStore


class FakeFastEmbed:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> Any:
        self.passages.extend(texts)
        return iter(np.full(3, float(len(t)), dtype=np.float32) for t in texts)

    def query_embed(self, texts: list[str]) -> Any:
        self.queries.extend(texts)
        return iter(np.zeros(3, dtype=np.float32) for _ in texts)


def test_local_embedder_batches_and_uses_query_mode() -> None:
    model = FakeFastEmbed()
    loads: list[tuple[str, str | None]] = []
    embedder = LocalEmbedder(
        "BAAI/bge-small-en-v1.5",
        cache_dir=Path("cache"),
        model_factory=lambda name, cache: loads.append((name, cache)) or model,
        dim_lookup=lambda name: 3,
    )

    assert embedder.dim == 3 and loads == []  # dim known without loading
    assert embedder.embed(["ab", "abcd"]) == [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    assert embedder.embed_query("q") == [0.0, 0.0, 0.0]
    assert embedder.embed([]) == []
    assert loads == [("BAAI/bge-small-en-v1.5", "cache")]  # loaded once
    assert model.queries == ["q"]


def test_local_embedder_load_failure_is_retrieval_error() -> None:
    def broken(name: str, cache: str | None) -> Any:
        raise OSError("no network")

    with pytest.raises(RetrievalError, match="no network"):
        LocalEmbedder(model_factory=broken, dim_lookup=lambda n: 3).embed(["x"])


def test_chroma_store_round_trip(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", collection="test")
    assert store.count() == 0
    assert store.query([1.0, 0.0, 0.0], 5) == []

    store.upsert(
        ["a::0", "b::0", "c::0"],
        [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]],
        ["alpha", "beta", "gamma"],
        [{"note_name": "a", "chunk_index": 0}, {"note_name": "b", "chunk_index": 0}, {"note_name": "c", "chunk_index": 0}],
    )
    hits = store.query([1.0, 0.0, 0.0], 2)

    assert [h.id for h in hits] == ["a::0", "b::0"]
    assert hits[0].document == "alpha" and hits[0].metadata["note_name"] == "a"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert 0.6 < hits[1].score < 0.8  # cosine(1,0,0 ; .7,.7,0) ≈ 0.707

    store.upsert(["a::0"], [[0.0, 0.0, 1.0]], ["alpha v2"], [{"note_name": "a", "chunk_index": 0}])
    assert store.count() == 3  # replaced, not duplicated
    store.delete(["b::0", "missing::0"])
    assert store.count() == 2
    assert store.query([0.0, 0.0, 1.0], 10)[0].document in {"alpha v2", "gamma"}

    store.reset()
    assert store.count() == 0
    store.upsert(["x::0"], [[1.0, 2.0]], ["new dim"], [{"note_name": "x", "chunk_index": 0}])
    assert store.count() == 1  # reset also cleared the old 3-d dimension


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(vault_path=tmp_path, _env_file=None, **kw)  # type: ignore[call-arg]


def test_factories_select_backends_without_loading(tmp_path: Path) -> None:
    settings = _settings(tmp_path, chroma_path=tmp_path / "data" / "chroma")
    embedder = create_embedder(settings)
    store = create_vector_store(settings)
    assert isinstance(embedder, LocalEmbedder) and isinstance(store, ChromaStore)
    assert not (tmp_path / "data" / "chroma").exists()  # nothing opened yet

    memory = build_memory(settings, tmp_path / "Jarvis")
    assert memory.indexer._manifest_path == tmp_path / "data" / "index_manifest.json"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("embedder_provider", "gemini", "not implemented"), ("embedder_provider", "openai", "unknown"),
     ("vector_store", "pgvector", "unknown")],
)
def test_unsupported_backends_are_rejected(tmp_path: Path, field: str, value: str, match: str) -> None:
    settings = _settings(tmp_path, **{field: value})
    with pytest.raises(RetrievalError, match=match):
        build_memory(settings, tmp_path)
