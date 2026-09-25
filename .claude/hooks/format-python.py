"""PostToolUse hook: run `ruff format` on the Python file Claude just wrote.

Reads the hook payload on stdin, extracts the touched path, and formats it in
place. Always exits 0 -- a formatting failure must never block an edit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    raw_path = tool_response.get("filePath") or tool_input.get("file_path") or ""
    if not raw_path.endswith(".py"):
        return 0

    target = Path(raw_path)
    if not target.is_file():
        return 0

    root = Path(__file__).resolve().parents[2]
    for candidate in (root / ".venv" / "Scripts" / "ruff.exe", root / ".venv" / "bin" / "ruff"):
        if candidate.is_file():
            cmd = [str(candidate), "format", "--", str(target)]
            break
    else:
        cmd = ["uv", "run", "--project", str(root), "ruff", "format", "--", str(target)]

    try:
        subprocess.run(cmd, cwd=root, capture_output=True, timeout=45)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
