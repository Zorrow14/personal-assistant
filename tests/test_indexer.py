"""VaultIndexer tests: tmp vault, fake embedder + in-memory store, no downloads."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.memory.documents import chunk_text, parse_note
from jarvis.memory.indexer import VaultIndexer
from jarvis.memory.vault import Vault
from tests.fakes import FakeEmbedder, FakeVectorStore

WHEN = datetime(2026, 9, 25, 18, 27)


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    root = tmp_path / "vault"
    root.mkdir()
    return Vault(root, clock=lambda: WHEN)


class Env:
    def __init__(self, vault: Vault, tmp_path: Path, **kwargs: object) -> None:
        self.vault = vault
        self.embedder = FakeEmbedder()
        self.store = FakeVectorStore()
        self.manifest = tmp_path / ".jarvis" / "index_manifest.json"
        self.kwargs = {"chunk_chars": 200, "chunk_overlap": 30, **kwargs}
        self.indexer = self.make()

    def make(self, **overrides: object) -> VaultIndexer:
        return VaultIndexer(
            self.vault.jarvis_root, self.embedder, self.store, self.manifest,
            **{**self.kwargs, **overrides},  # type: ignore[arg-type]
        )


@pytest.fixture
def env(vault: Vault, tmp_path: Path) -> Env:
    return Env(vault, tmp_path)


def _touch_later(path: Path) -> None:
    """Bump mtime so a same-size rewrite still looks changed."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_notes_become_chunks_with_stable_ids_and_metadata(env: Env) -> None:
    note = env.vault.write_task_note(
        "Finish the wake-word module", "Tuned the threshold.", tags=["coding", "voice"]
    )
    env.vault.append_daily("Started working on memory")

    stats = env.indexer.reindex_all()

    assert (stats.added, stats.updated, stats.skipped, stats.removed) == (2, 0, 0, 0)
    task_id = f"Tasks/{note.name}::0"
    assert set(env.store.records) == {task_id, "Daily/2026-09-25.md::0"}
    _, document, meta = env.store.records[task_id]
    assert document.startswith("Finish the wake-word module\n# Finish the wake-word module")
    assert meta == {
        "source_path": f"Tasks/{note.name}",
        "note_name": note.stem,
        "date": "2026-09-25",
        "tags": "coding, voice",
        "command": "Finish the wake-word module",
        "chunk_index": 0,
    }
    daily_meta = env.store.records["Daily/2026-09-25.md::0"][2]
    assert daily_meta["date"] == "2026-09-25"  # from the filename: daily notes have no frontmatter
    assert daily_meta["command"] == ""


def test_long_note_is_chunked_and_reindex_replaces_not_duplicates(env: Env) -> None:
    body = " ".join(f"sentence number {i} about the vault module." for i in range(40))
    note = env.vault.write_task_note("Write the long note", body)

    first = env.indexer.reindex_all()
    ids = sorted(env.store.records)
    assert first.chunks == len(ids) > 3
    assert all(i.startswith(f"Tasks/{note.name}::") for i in ids)

    env.manifest.unlink()  # force a full re-embed
    env.indexer.reindex_all()
    assert sorted(env.store.records) == ids  # same ids, replaced in place


def test_unchanged_notes_are_skipped_on_second_pass(env: Env) -> None:
    env.vault.write_task_note("One", "a")
    env.vault.write_task_note("Two", "b")
    env.indexer.reindex_all()
    embedded = env.embedder.texts_embedded

    stats = env.indexer.reindex_all()

    assert (stats.added, stats.updated, stats.skipped) == (0, 0, 2)
    assert env.embedder.texts_embedded == embedded  # nothing re-embedded


def test_changed_note_is_updated_and_stale_chunks_removed(env: Env) -> None:
    long_body = " ".join(f"word{i}" for i in range(200))
    note = env.vault.write_task_note("Shrinking note", long_body)
    env.indexer.reindex_all()
    before = len(env.store.records)
    assert before > 1

    note.write_text("---\ncommand: \"Shrinking note\"\n---\nNow short.\n", encoding="utf-8")
    stats = env.indexer.reindex_all()

    assert stats.updated == 1
    assert list(env.store.records) == [f"Tasks/{note.name}::0"]
    assert "Now short." in env.store.records[f"Tasks/{note.name}::0"][1]


def test_deleted_note_chunks_are_dropped(env: Env) -> None:
    keep = env.vault.write_task_note("Keep me", "x")
    gone = env.vault.write_task_note("Delete me", "y")
    env.indexer.reindex_all()

    gone.unlink()
    stats = env.indexer.reindex_all()

    assert stats.removed == 1
    assert all(k.startswith(f"Tasks/{keep.name}") for k in env.store.records)
    assert f"Tasks/{gone.name}::0" in env.store.deleted


def test_index_paths_indexes_only_given_notes(env: Env, tmp_path: Path) -> None:
    a = env.vault.write_task_note("Alpha", "first")
    b = env.vault.write_task_note("Beta", "second")
    outside = tmp_path / "elsewhere.md"
    outside.write_text("not in the vault", encoding="utf-8")

    stats = env.indexer.index_paths([a, outside, a.with_suffix(".txt")])

    assert stats.added == 1
    assert set(env.store.records) == {f"Tasks/{a.name}::0"}
    env.indexer.index_paths([b])
    assert len(env.store.records) == 2
    assert env.indexer.indexed_notes() == 2


def test_index_paths_removes_missing_file(env: Env) -> None:
    note = env.vault.write_task_note("Temp", "x")
    env.indexer.index_paths([note])
    note.unlink()

    stats = env.indexer.index_paths([note])

    assert stats.removed == 1
    assert env.store.count() == 0


def test_empty_store_with_stale_manifest_rebuilds(env: Env) -> None:
    env.vault.write_task_note("Survivor", "x")
    env.indexer.reindex_all()
    env.store.records.clear()  # e.g. the .jarvis/chroma folder was deleted

    stats = env.indexer.reindex_all()

    assert stats.added == 1 and env.store.count() == 1


def test_changing_model_or_chunking_resets_and_rebuilds(env: Env) -> None:
    env.vault.write_task_note("Note", "x")
    env.indexer.reindex_all()
    resets = env.store.resets

    stats = env.make(index_key="local:another-model").reindex_all()

    assert env.store.resets == resets + 1
    assert stats.added == 1  # re-embedded, not skipped


def test_snapshot_detects_new_modified_and_deleted(env: Env) -> None:
    existing = env.vault.write_task_note("Existing", "x")
    doomed = env.vault.write_task_note("Doomed", "y")
    snap = env.indexer.snapshot()

    new = env.vault.write_task_note("New", "z")
    existing.write_text(existing.read_text(encoding="utf-8") + "more\n", encoding="utf-8")
    _touch_later(existing)
    doomed.unlink()

    changed = {p.name for p in env.indexer.changed_since(snap)}
    assert changed == {new.name, existing.name, doomed.name}


def test_parse_note_round_trips_writer_output(vault: Vault) -> None:
    path = vault.write_task_note(
        'remind me: call "Mum"', "Body text.", tags=["family", "a b"], tools_used=["phone"],
        links=["Mum"],
    )
    note = parse_note(path.read_text(encoding="utf-8"))
    assert note.frontmatter == {
        "date": "2026-09-25",
        "command": 'remind me: call "Mum"',
        "status": "completed",
        "tags": ["family", "a b"],
        "tools_used": ["phone"],
    }
    assert note.body.startswith("# Remind me")
    assert note.body.endswith("Related: [[Mum]]")


def test_parse_note_tolerates_notes_without_or_with_broken_frontmatter() -> None:
    assert parse_note("# Just a heading\ntext").frontmatter == {}
    broken = parse_note("---\ncommand: x\nno closing fence")
    assert broken.frontmatter == {} and "no closing fence" in broken.body
    crlf = parse_note("﻿---\r\ncommand: \"hi\"\r\n---\r\nbody\r\n")
    assert crlf.frontmatter == {"command": "hi"} and crlf.body == "body"


def test_chunk_text_respects_size_and_overlap() -> None:
    text = " ".join(f"w{i:03d}" for i in range(300))  # 1499 chars
    chunks = chunk_text(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # overlapping boundary
    assert " ".join(chunks).count("w299") >= 1
    assert chunk_text("  short  ", 200, 40) == ["short"]
    assert chunk_text("   ", 200, 40) == []
    with pytest.raises(ValueError):
        chunk_text("x", 100, 100)
