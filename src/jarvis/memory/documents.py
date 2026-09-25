"""Reading vault notes for indexing: frontmatter parsing and text chunking.

Pure text helpers, with no embedding or storage knowledge. The frontmatter parser
handles the subset of YAML that `vault.py` writes (scalars, JSON-quoted
strings and flat `[a, b]` lists) and ignores anything it doesn't understand,
so hand-written notes never break indexing.
"""

import json
import re
from dataclasses import dataclass, field

_KEY_VALUE = re.compile(r"^([A-Za-z_][\w-]*):\s*(.*)$")
_LIST_ITEM = re.compile(r'"(?:[^"\\]|\\.)*"|[^,]+')

FrontmatterValue = str | list[str]


@dataclass(frozen=True)
class ParsedNote:
    """A note split into its frontmatter fields and Markdown body."""

    frontmatter: dict[str, FrontmatterValue] = field(default_factory=dict)
    body: str = ""


def parse_note(text: str) -> ParsedNote:
    """Split a Markdown note into frontmatter and body.

    A note without a leading `---` block is all body.
    """
    text = text.lstrip("﻿")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ParsedNote(body=text.strip())
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return ParsedNote(body=text.strip())  # unterminated: treat as plain text

    frontmatter: dict[str, FrontmatterValue] = {}
    for line in lines[1:end]:
        match = _KEY_VALUE.match(line.strip())
        if match:
            frontmatter[match.group(1)] = _parse_value(match.group(2).strip())
    body = "\n".join(lines[end + 1 :]).strip()
    return ParsedNote(frontmatter=frontmatter, body=body)


def _parse_value(raw: str) -> FrontmatterValue:
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        items = [_parse_scalar(m.group(0).strip()) for m in _LIST_ITEM.finditer(inner)]
        return [item for item in items if item]
    return _parse_scalar(raw)


def _parse_scalar(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        try:
            return str(json.loads(raw))
        except json.JSONDecodeError:
            return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1].replace("''", "'")
    return raw


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split `text` into chunks of at most `max_chars`, overlapping by about `overlap`.

    Cuts prefer a line break, then a space, in the back half of each window so
    words and lines aren't split mid-way.

    Raises:
        ValueError: If `overlap` is not smaller than `max_chars`.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            floor = start + max_chars // 2
            cut = text.rfind("\n", floor, end)
            if cut == -1:
                cut = text.rfind(" ", floor, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
