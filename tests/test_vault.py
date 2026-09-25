"""Tests for the Obsidian vault writer. Uses tmp_path; never touches a real vault."""

from datetime import datetime
from pathlib import Path

import pytest

from jarvis.memory.vault import Vault, VaultError, VaultPathError, slugify

FIXED_NOW = datetime(2026, 9, 25, 14, 30)


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    root = tmp_path / "vault"
    root.mkdir()
    return Vault(root, clock=lambda: FIXED_NOW)


def _split_frontmatter(text: str) -> tuple[list[str], str]:
    assert text.startswith("---\n"), "note must open with a frontmatter fence"
    frontmatter, sep, rest = text[len("---\n") :].partition("\n---\n")
    assert sep, "frontmatter must be closed with a --- fence"
    return frontmatter.splitlines(), rest


def test_write_task_note_produces_frontmatter_and_body(vault: Vault) -> None:
    path = vault.write_task_note(
        "Turn on the living room lights",
        "Switched on 3 lights.",
        tags=["home", "lights"],
        tools_used=["smart_home"],
        links=["Living Room", "[[Lights]]"],
    )

    assert path.parent == vault.jarvis_root / "Tasks"
    assert path.name == "2026-09-25-1430-turn-on-the-living-room-lights.md"

    frontmatter, rest = _split_frontmatter(path.read_text(encoding="utf-8"))
    assert frontmatter == [
        "date: 2026-09-25",
        'command: "Turn on the living room lights"',
        "status: completed",
        "tags: [home, lights]",
        "tools_used: [smart_home]",
    ]
    assert rest == (
        "\n"
        "# Turn on the living room lights\n"
        "\n"
        "Switched on 3 lights.\n"
        "\n"
        "Related: [[Living Room]], [[Lights]]\n"
    )


def test_write_task_note_defaults_and_quoting(vault: Vault) -> None:
    path = vault.write_task_note('remind me: call "Mum"', "Done.", status="failed")

    frontmatter, rest = _split_frontmatter(path.read_text(encoding="utf-8"))
    assert 'command: "remind me: call \\"Mum\\""' in frontmatter
    assert "status: failed" in frontmatter
    assert "tags: []" in frontmatter
    assert "tools_used: []" in frontmatter
    assert rest.startswith("\n# Remind me: call \"Mum\"\n")
    assert "Related:" not in rest


def test_write_task_note_never_overwrites(vault: Vault) -> None:
    first = vault.write_task_note("check weather", "Sunny.")
    second = vault.write_task_note("check weather", "Still sunny.")

    assert first != second
    assert second.name == "2026-09-25-1430-check-weather-2.md"
    assert "Sunny." in first.read_text(encoding="utf-8")


def test_write_task_note_rejects_blank_command(vault: Vault) -> None:
    with pytest.raises(ValueError):
        vault.write_task_note("   ", "body")


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.md",
        "../../outside.md",
        "Tasks/../../outside.md",
        "Tasks/../../vault-sibling/evil.md",
    ],
)
def test_safe_path_rejects_relative_escape(vault: Vault, escape: str) -> None:
    with pytest.raises(VaultPathError):
        vault._safe_path(escape)


def test_safe_path_rejects_absolute_path_outside(vault: Vault, tmp_path: Path) -> None:
    with pytest.raises(VaultPathError):
        vault._safe_path(tmp_path / "elsewhere.md")
    with pytest.raises(VaultPathError):
        vault._safe_path(vault.vault_root / "NotJarvis" / "note.md")


def test_safe_path_accepts_paths_inside_jarvis(vault: Vault) -> None:
    resolved = vault._safe_path("Tasks/sub/../note.md")
    assert resolved == vault.jarvis_root / "Tasks" / "note.md"


def test_traversal_in_command_cannot_escape_tasks_folder(vault: Vault, tmp_path: Path) -> None:
    path = vault.write_task_note("../../../../etc/passwd", "nope")

    assert path.parent == vault.jarvis_root / "Tasks"
    assert ".." not in path.name
    written = [p.resolve() for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [path]


def test_slugify() -> None:
    assert slugify("  Hello, World!  ") == "hello-world"
    assert slugify("../../x") == "x"
    assert slugify("!!!") == "task"
    assert len(slugify("word " * 50)) <= 60


def test_append_daily_creates_then_appends(vault: Vault) -> None:
    first = vault.append_daily("Started the day")
    second = vault.append_daily("Checked   the\nweather")

    assert first == second == vault.jarvis_root / "Daily" / "2026-09-25.md"
    assert first.read_text(encoding="utf-8") == (
        "# 2026-09-25\n\n- 14:30 Started the day\n- 14:30 Checked the weather\n"
    )


def test_search_vault_matches_filenames(vault: Vault) -> None:
    lights = vault.write_task_note("turn on the lights", "ok")
    vault.write_task_note("check weather", "ok")

    assert vault.search_vault("LIGHTS") == [lights]
    assert vault.search_vault("turn on") == [lights]
    assert vault.search_vault("nothing-matches") == []
    assert vault.search_vault("   ") == []


def test_vault_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(VaultError):
        Vault(tmp_path / "does-not-exist")


def test_check_writable_creates_jarvis_folder(vault: Vault) -> None:
    assert vault.check_writable() == vault.jarvis_root
    assert vault.jarvis_root.is_dir()
    assert list(vault.jarvis_root.iterdir()) == []
