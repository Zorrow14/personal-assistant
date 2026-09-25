"""Embedded, on-disk vector store with Chroma. The only module that knows about chromadb.

Embeddings are always computed by our `Embedder` and passed in; the collection
is created without an embedding function, so Chroma never embeds (or downloads
a model) itself. Telemetry is switched off.
"""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.core.interfaces import MetadataValue, RetrievalError, SearchHit, VectorStore
from jarvis.logging import get_logger

DEFAULT_COLLECTION = "jarvis_vault"
_UPSERT_BATCH = 500

ClientFactory = Callable[[Path], Any]

log = get_logger(__name__)


def _persistent_client(path: Path) -> Any:
    import chromadb  # heavy import; only when first needed
    from chromadb.config import Settings as ChromaSettings

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


class ChromaStore(VectorStore):
    """`VectorStore` backed by a persistent Chroma collection (cosine distance)."""

    def __init__(
        self,
        path: Path,
        collection: str = DEFAULT_COLLECTION,
        *,
        client_factory: ClientFactory = _persistent_client,
    ) -> None:
        """
        Args:
            path: Directory holding the Chroma database (created if missing).
            collection: Collection name.
            client_factory: Builds the Chroma client (injectable for tests).
        """
        self._path = Path(path)
        self._collection_name = collection
        self._client_factory = client_factory
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.Lock()

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, MetadataValue]],
    ) -> None:
        """Insert or replace records, in batches Chroma accepts."""
        collection = self._get_collection()
        for start in range(0, len(ids), _UPSERT_BATCH):
            end = start + _UPSERT_BATCH
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def query(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """Nearest records to `embedding`, best first, scored as cosine similarity."""
        collection = self._get_collection()
        available = collection.count()
        if available == 0 or top_k <= 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            SearchHit(
                id=hit_id,
                document=documents[i] or "",
                metadata=dict(metadatas[i] or {}),
                # Cosine distance is 1 - similarity.
                score=max(0.0, min(1.0, 1.0 - float(distances[i]))),
            )
            for i, hit_id in enumerate(ids)
        ]

    def delete(self, ids: list[str]) -> None:
        """Remove records by id."""
        if ids:
            self._get_collection().delete(ids=ids)

    def count(self) -> int:
        """Number of stored chunks."""
        return int(self._get_collection().count())

    def reset(self) -> None:
        """Drop and recreate the collection (clears the fixed vector dimension too)."""
        with self._lock:
            client = self._get_client()
            try:
                client.delete_collection(self._collection_name)
            except Exception:  # the collection may not exist yet
                pass
            self._collection = None
        log.info("vector_store.reset", collection=self._collection_name)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = self._client_factory(self._path)
            except Exception as exc:
                raise RetrievalError(f"could not open the Chroma index at {self._path}: {exc}") from exc
        return self._client

    def _get_collection(self) -> Any:
        with self._lock:
            if self._collection is None:
                self._collection = self._get_client().get_or_create_collection(
                    name=self._collection_name,
                    embedding_function=None,  # we always supply our own vectors
                    configuration={"hnsw": {"space": "cosine"}},
                )
                log.debug("vector_store.opened", path=str(self._path), collection=self._collection_name)
            return self._collection


# TODO(phase-later): PgVectorStore(VectorStore), a swap-in once there's a server to host it.
