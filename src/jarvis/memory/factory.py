"""Build the retrieval stack (embedder, vector store, indexer) from settings."""

from dataclasses import dataclass
from pathlib import Path

from jarvis.config import Settings
from jarvis.core.interfaces import Embedder, RetrievalError, VectorStore
from jarvis.memory.indexer import VaultIndexer

MANIFEST_NAME = "index_manifest.json"
MODEL_CACHE_DIR = "fastembed"


@dataclass(frozen=True)
class MemoryStack:
    """The pieces retrieval needs, built once and shared."""

    embedder: Embedder
    store: VectorStore
    indexer: VaultIndexer


def create_embedder(settings: Settings) -> Embedder:
    """Build the Embedder named by `settings.embedder_provider`. Nothing is loaded yet.

    Raises:
        RetrievalError: If the provider is unknown or not implemented.
    """
    if settings.embedder_provider == "local":
        from jarvis.memory.embedder import LocalEmbedder

        return LocalEmbedder(settings.embed_model, cache_dir=settings.data_dir / MODEL_CACHE_DIR)
    if settings.embedder_provider == "gemini":
        # TODO(phase-later): GeminiEmbedder, an opt-in that sends note text to Google.
        raise RetrievalError("embedder_provider 'gemini' is not implemented yet; use 'local'")
    raise RetrievalError(f"unknown embedder_provider {settings.embedder_provider!r}; supported: local")


def create_vector_store(settings: Settings) -> VectorStore:
    """Build the VectorStore named by `settings.vector_store`. Nothing is opened yet.

    Raises:
        RetrievalError: If the backend is unknown.
    """
    if settings.vector_store == "chroma":
        from jarvis.memory.vector_store import ChromaStore

        return ChromaStore(settings.chroma_path.expanduser())
    # TODO(phase-later): "pgvector" -> PgVectorStore.
    raise RetrievalError(f"unknown vector_store {settings.vector_store!r}; supported: chroma")


def build_memory(settings: Settings, jarvis_root: Path) -> MemoryStack:
    """Wire embedder + store + indexer for the vault's `Jarvis/` folder.

    Raises:
        RetrievalError: If a backend is unknown.
    """
    embedder = create_embedder(settings)
    store = create_vector_store(settings)
    indexer = VaultIndexer(
        jarvis_root,
        embedder,
        store,
        settings.data_dir / MANIFEST_NAME,
        chunk_chars=settings.rag_chunk_chars,
        chunk_overlap=settings.rag_chunk_overlap,
        index_key=f"{settings.embedder_provider}:{settings.embed_model}",
    )
    return MemoryStack(embedder, store, indexer)
