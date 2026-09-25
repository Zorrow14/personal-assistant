"""Vault indexing for retrieval: notes -> chunks -> embeddings -> vector store.

Incremental. A JSON manifest records each note's content hash and chunk count, so
unchanged notes are skipped, changed notes are re-embedded (stale chunks
removed), and deleted notes' chunks are dropped. Chunk ids are
`<path relative to Jarvis/>::<chunk index>`, so re-indexing replaces rather than
duplicates. If the embedding model, chunking settings or vault change, the
manifest no longer matches and the index is rebuilt from scratch.
"""

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.core.interfaces import Embedder, MetadataValue, VectorStore
from jarvis.logging import get_logger
from jarvis.memory.documents import chunk_text, parse_note

NOTE_GLOB = "*.md"
MANIFEST_VERSION = 1
_DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})")

Snapshot = dict[str, tuple[int, int]]
"""Relative note path -> (mtime_ns, size): a cheap "what's on disk" fingerprint."""

log = get_logger(__name__)


@dataclass
class IndexStats:
    """What an indexing pass did, counted in notes (chunks where noted)."""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    """Chunks embedded and written during this pass."""

    @property
    def changed(self) -> bool:
        """True if anything was written or removed."""
        return bool(self.added or self.updated or self.removed)


class VaultIndexer:
    """Keeps a vector index in sync with the Markdown notes under `<vault>/Jarvis/`."""

    def __init__(
        self,
        jarvis_root: Path,
        embedder: Embedder,
        store: VectorStore,
        manifest_path: Path,
        *,
        chunk_chars: int = 1000,
        chunk_overlap: int = 150,
        index_key: str = "",
    ) -> None:
        """
        Args:
            jarvis_root: The vault's `Jarvis/` folder (Tasks, Daily, ...).
            embedder: Turns chunks into vectors.
            store: Where vectors live.
            manifest_path: JSON file tracking what has been indexed.
            chunk_chars: Maximum characters per chunk.
            chunk_overlap: Characters shared by consecutive chunks.
            index_key: Identifies the embedding model; a change forces a rebuild.
        """
        self.jarvis_root = Path(jarvis_root).resolve()
        self._embedder = embedder
        self._store = store
        self._manifest_path = Path(manifest_path)
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap
        self._index_key = index_key
        self._fingerprint_cache: str | None = None

    @property
    def _fingerprint(self) -> str:
        """Everything that, if changed, makes existing vectors unusable. Lazy: `dim`
        may import the embedding library, which startup shouldn't pay for."""
        if self._fingerprint_cache is None:
            parts = [
                MANIFEST_VERSION,
                self._index_key,
                self._embedder.dim,
                self._chunk_chars,
                self._chunk_overlap,
                self.jarvis_root,
            ]
            self._fingerprint_cache = "|".join(str(p) for p in parts)
        return self._fingerprint_cache

    # -- public API -----------------------------------------------------------

    def reindex_all(self) -> IndexStats:
        """Bring the whole index up to date (incremental: unchanged notes are skipped)."""
        files = self._load_manifest()
        stats = IndexStats()
        on_disk = {self._relative(path) for path in self._note_paths()}
        for rel in sorted(on_disk):
            self._index_one(rel, files, stats)
        for rel in sorted(set(files) - on_disk):
            self._remove(rel, files, stats)
        self._save_manifest(files)
        self._log("reindex_all", stats)
        return stats

    def index_paths(self, paths: Iterable[Path]) -> IndexStats:
        """Index specific notes (e.g. just written); missing files are removed from the index.

        Paths outside `Jarvis/` or not ending in `.md` are ignored.
        """
        files = self._load_manifest()
        stats = IndexStats()
        for path in paths:
            rel = self._relative_or_none(Path(path))
            if rel is None:
                log.debug("indexer.path_ignored", path=str(path))
                continue
            if (self.jarvis_root / rel).is_file():
                self._index_one(rel, files, stats)
            else:
                self._remove(rel, files, stats)
        self._save_manifest(files)
        if stats.changed:
            self._log("index_paths", stats)
        return stats

    def snapshot(self) -> Snapshot:
        """Fingerprint every note (mtime + size) without reading contents."""
        snap: Snapshot = {}
        for path in self._note_paths():
            try:
                st = path.stat()
            except OSError:
                continue
            snap[self._relative(path)] = (st.st_mtime_ns, st.st_size)
        return snap

    def changed_since(self, before: Snapshot) -> list[Path]:
        """Notes created, modified or deleted since `before` was taken."""
        after = self.snapshot()
        changed = {rel for rel, sig in after.items() if before.get(rel) != sig}
        changed |= set(before) - set(after)
        return [self.jarvis_root / rel for rel in sorted(changed)]

    def indexed_notes(self) -> int:
        """How many notes the manifest lists (cheap: opens neither the store nor the model)."""
        files = self._read_manifest().get("files", {})
        return len(files) if isinstance(files, dict) else 0

    # -- internals ------------------------------------------------------------

    def _index_one(self, rel: str, files: dict[str, dict[str, Any]], stats: IndexStats) -> None:
        path = self.jarvis_root / rel
        try:
            raw = path.read_bytes()
        except OSError as exc:
            log.warning("indexer.read_failed", path=rel, error=str(exc))
            return
        digest = hashlib.sha256(raw).hexdigest()
        previous = files.get(rel)
        if previous is not None and previous.get("hash") == digest:
            stats.skipped += 1
            return

        ids, documents, metadatas = self._chunks_for(rel, raw.decode("utf-8", errors="replace"))
        if ids:
            embeddings = self._embedder.embed(documents)
            self._store.upsert(ids, embeddings, documents, metadatas)
        old_count = int(previous.get("chunks", 0)) if previous else 0
        stale = [f"{rel}::{i}" for i in range(len(ids), old_count)]
        self._store.delete(stale)

        files[rel] = {"hash": digest, "chunks": len(ids)}
        stats.chunks += len(ids)
        if previous is None:
            stats.added += 1
        else:
            stats.updated += 1

    def _remove(self, rel: str, files: dict[str, dict[str, Any]], stats: IndexStats) -> None:
        previous = files.pop(rel, None)
        if previous is None:
            return
        count = int(previous.get("chunks", 0))
        self._store.delete([f"{rel}::{i}" for i in range(count)])
        stats.removed += 1

    def _chunks_for(
        self, rel: str, text: str
    ) -> tuple[list[str], list[str], list[dict[str, MetadataValue]]]:
        note = parse_note(text)
        command = _as_text(note.frontmatter.get("command"))
        # Put the command first so it's searchable even when the body is terse.
        content = f"{command}\n{note.body}" if command else note.body
        chunks = chunk_text(content, self._chunk_chars, self._chunk_overlap)

        stem = Path(rel).stem
        date = _as_text(note.frontmatter.get("date"))
        if not date:
            match = _DATE_PREFIX.match(stem)
            date = match.group(1) if match else ""
        base: dict[str, MetadataValue] = {
            "source_path": rel,
            "note_name": stem,
            "date": date,
            "tags": ", ".join(_as_list(note.frontmatter.get("tags"))),
            "command": command,
        }
        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        metadatas = [{**base, "chunk_index": i} for i in range(len(chunks))]
        return ids, chunks, metadatas

    def _note_paths(self) -> list[Path]:
        if not self.jarvis_root.is_dir():
            return []
        return sorted(p for p in self.jarvis_root.rglob(NOTE_GLOB) if p.is_file())

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.jarvis_root).as_posix()

    def _relative_or_none(self, path: Path) -> str | None:
        if path.suffix.lower() != ".md":
            return None
        try:
            return self._relative(path)
        except ValueError:
            return None

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """The manifest's file table, or a fresh one (after resetting the store) if stale."""
        data = self._read_manifest()
        if data.get("fingerprint") != self._fingerprint:
            if data:
                log.info("indexer.rebuild", reason="embedding model, chunking or vault changed")
            self._store.reset()
            return {}
        files: dict[str, dict[str, Any]] = data.get("files", {})
        if files and self._store.count() == 0:
            log.info("indexer.rebuild", reason="index is empty but manifest is not")
            return {}
        return files

    def _read_manifest(self) -> dict[str, Any]:
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_manifest(self, files: dict[str, dict[str, Any]]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fingerprint": self._fingerprint, "files": files}, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self._manifest_path)

    def _log(self, operation: str, stats: IndexStats) -> None:
        log.info(
            f"indexer.{operation}",
            added=stats.added,
            updated=stats.updated,
            skipped=stats.skipped,
            removed=stats.removed,
            chunks=stats.chunks,
        )


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value).strip() if value else ""


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [value] if isinstance(value, str) and value.strip() else []
