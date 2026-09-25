"""Obsidian vault writer.

Everything Jarvis writes lives under `<vault_path>/Jarvis/`. Every path is
resolved through `Vault._safe_path`, which refuses anything outside that subtree.
"""

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from jarvis.logging import get_logger

JARVIS_DIR = "Jarvis"
TASKS_DIR = "Tasks"
DAILY_DIR = "Daily"
NOTE_SUFFIX = ".md"
MAX_SLUG_LENGTH = 60
_MAX_NAME_COLLISIONS = 1000

_PLAIN_YAML_SCALAR = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")
_YAML_KEYWORDS = frozenset({"true", "false", "yes", "no", "on", "off", "null"})

log = get_logger(__name__)


class VaultError(Exception):
    """A vault operation failed."""


class VaultPathError(VaultError):
    """A target path would land outside `<vault_path>/Jarvis/`."""


def slugify(text: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """Reduce `text` to a filesystem-safe `lowercase-hyphenated` slug.

    Only ASCII letters and digits survive, so separators and `..` can never
    reach the filename. Falls back to "task" if nothing is left.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "task"


class Vault:
    """Reads and writes Jarvis notes inside an Obsidian vault."""

    def __init__(self, vault_path: Path, *, clock: Callable[[], datetime] = datetime.now) -> None:
        """
        Args:
            vault_path: Root of an existing Obsidian vault.
            clock: Source of the current time (injectable for tests).

        Raises:
            VaultError: If `vault_path` is not an existing directory.
        """
        root = Path(vault_path).expanduser()
        if not root.is_dir():
            raise VaultError(f"vault path is not an existing directory: {root}")
        self._vault_root = root.resolve()
        self._jarvis_root = self._vault_root / JARVIS_DIR
        self._clock = clock

    @property
    def vault_root(self) -> Path:
        """Resolved root of the Obsidian vault."""
        return self._vault_root

    @property
    def jarvis_root(self) -> Path:
        """The `Jarvis/` subtree that all writes are confined to."""
        return self._jarvis_root

    def write_task_note(
        self,
        command: str,
        body: str,
        *,
        status: str = "completed",
        tags: Sequence[str] | None = None,
        tools_used: Sequence[str] | None = None,
        links: Sequence[str] | None = None,
    ) -> Path:
        """Journal a task as `Jarvis/Tasks/YYYY-MM-DD-HHMM-<slug>.md`.

        A numeric suffix (`-2`, `-3`, ...) is added rather than overwriting an
        existing note.

        Returns:
            Path of the new note.

        Raises:
            ValueError: If `command` is blank.
        """
        if not command.strip():
            raise ValueError("command must not be blank")
        now = self._clock()
        content = _render_task_note(
            command,
            body,
            date=now,
            status=status,
            tags=tags or (),
            tools_used=tools_used or (),
            links=links or (),
        )
        path = self._write_new_note(Path(TASKS_DIR), f"{now:%Y-%m-%d-%H%M}-{slugify(command)}", content)
        log.info("vault.task_note_written", path=str(path), status=status)
        return path

    def append_daily(self, text: str) -> Path:
        """Append a `- HH:MM <text>` line to `Jarvis/Daily/YYYY-MM-DD.md`.

        Returns:
            Path of the daily note.
        """
        now = self._clock()
        path = self._safe_path(Path(DAILY_DIR) / f"{now:%Y-%m-%d}{NOTE_SUFFIX}")
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            if is_new:
                fh.write(f"# {now:%Y-%m-%d}\n\n")
            fh.write(f"- {now:%H:%M} {_one_line(text)}\n")
        log.debug("vault.daily_appended", path=str(path))
        return path

    def search_vault(self, query: str) -> list[Path]:
        """Return Jarvis notes whose filename contains `query` (case-insensitive).

        Spaces in the query also match hyphens, so "turn on" finds `turn-on-...`.
        """
        # TODO(phase-4): semantic search
        needle = query.strip().lower()
        if not needle or not self._jarvis_root.is_dir():
            return []
        needles = {needle, needle.replace(" ", "-")}
        return sorted(
            path
            for path in self._jarvis_root.rglob(f"*{NOTE_SUFFIX}")
            if any(n in path.stem.lower() for n in needles)
        )

    def check_writable(self) -> Path:
        """Create `Jarvis/` if needed and prove it accepts writes.

        Returns:
            The Jarvis root.

        Raises:
            OSError: If the probe file cannot be written or removed.
        """
        probe = self._safe_path(".jarvis-health-probe")
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return self._jarvis_root

    def _safe_path(self, relative: str | Path) -> Path:
        """Resolve `relative` under `Jarvis/`, refusing anything that escapes it.

        Resolution follows `..` segments and symlinks, and absolute inputs are
        checked too, so the returned path is guaranteed to be inside the subtree.

        Raises:
            VaultPathError: If the resolved path is outside `<vault>/Jarvis/`.
        """
        target = (self._jarvis_root / relative).resolve()
        if not target.is_relative_to(self._jarvis_root):
            log.warning("vault.path_rejected", requested=str(relative), resolved=str(target))
            raise VaultPathError(f"refusing path outside {self._jarvis_root}: {relative}")
        return target

    def _write_new_note(self, folder: Path, stem: str, content: str) -> Path:
        """Create `folder/stem.md` without overwriting, adding a suffix on collision."""
        for attempt in range(1, _MAX_NAME_COLLISIONS + 1):
            name = stem if attempt == 1 else f"{stem}-{attempt}"
            path = self._safe_path(folder / f"{name}{NOTE_SUFFIX}")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x", encoding="utf-8", newline="\n") as fh:
                    fh.write(content)
            except FileExistsError:
                continue
            return path
        raise VaultError(f"too many notes named {stem!r} in {folder}")


def _render_task_note(
    command: str,
    body: str,
    *,
    date: datetime,
    status: str,
    tags: Sequence[str],
    tools_used: Sequence[str],
    links: Sequence[str],
) -> str:
    lines = [
        "---",
        f"date: {date:%Y-%m-%d}",
        f"command: {json.dumps(_one_line(command), ensure_ascii=False)}",
        f"status: {_yaml_scalar(status)}",
        f"tags: {_yaml_list(tags)}",
        f"tools_used: {_yaml_list(tools_used)}",
        "---",
        "",
        f"# {_title(command)}",
        "",
        body.strip(),
    ]
    if links:
        lines += ["", "Related: " + ", ".join(f"[[{link.strip('[] ')}]]" for link in links)]
    return "\n".join(lines) + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _title(command: str) -> str:
    text = _one_line(command)
    return text[:1].upper() + text[1:]


def _yaml_scalar(value: str) -> str:
    """Emit `value` bare when safe, otherwise as a double-quoted YAML string."""
    if _PLAIN_YAML_SCALAR.fullmatch(value) and value.lower() not in _YAML_KEYWORDS:
        return value
    # JSON strings are valid YAML double-quoted scalars.
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_yaml_scalar(v) for v in values) + "]"
