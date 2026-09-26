# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything runs through `uv` — deps live in a uv-managed `.venv`, so bare `python`/`pytest` will use the wrong interpreter.

- `uv sync` — install deps (Python >=3.12, pinned 3.12 via `.python-version`)
- `uv run pytest` — full suite; fully offline (fakes for LLM/STT/TTS/wake/VAD/memory, temp vault, no mic)
- `uv run pytest -k test_name` — single test
- `uv run python -m jarvis.cli` — text chat. Flags: `--voice`, `--wake`, `--serve` (local web panel; combines only with `--wake`), `--reindex`, `--health`, `--list-tools`, `--list-devices`, `--metrics`
- `uv run python eval/run.py --fake` — offline tool-selection eval (scripted LLM, temp vault); drop `--fake` to run it against the real model
- `uv run python -m jarvis.cli --list-tools` — the fastest check that a new tool registered (shows a SKIPPED section with the reason when discovery fails)
- `uv run ruff format .` / `uv run ruff check .` — format and lint (ruff is the only linter here; there is no typechecker and no CI)

A `PostToolUse` hook (`.claude/settings.json`) runs `ruff format` on every `.py` file Claude writes, so formatting is already handled. Most of the pre-existing code predates ruff and is not yet formatted — don't reformat files you aren't otherwise editing.

## Adding or changing a tool

Tools are auto-discovered — adding a capability means adding **one new file** in `src/jarvis/tools/`, with no registry to edit. Read the full contract before writing one:

@docs/writing-a-tool.md

Key points that are easy to get wrong:

- `discover_tool_classes` skips modules and classes whose name starts with `_`, abstract classes, and classes merely imported into a module. Duplicate `Tool.name` is a fatal `DuplicateToolError`.
- Any tool with a real-world side effect **must** set `requires_confirmation = True`. That ClassVar is the single source of truth for the CLI confirmation gate in `core/agent.py`; nothing else marks a tool as dangerous.
- Filesystem access must go through the sandbox guard — resolve the path, then check `is_relative_to(root)` (see `file_tools.resolve_in_sandbox` and `Vault._safe_path`). Writes outside `<vault>/Jarvis/` raise `VaultPathError`.
- Inject dependencies by overriding `from_context(cls, context: ToolContext)`; new *kinds* of dependency get a field on `ToolContext` (`tools/context.py`). All wiring happens in `cli.py::build_tool_registry` / `build_agent` — the only place the object graph is assembled.
- Wrap blocking work in `await asyncio.to_thread(...)`. Return short text and don't raise for expected failures — the agent feeds tool errors back to the model rather than crashing.
- Keep third-party imports inside your own module, ideally lazy. `core/interfaces.py` holds provider-neutral ABCs; `llm/gemini_client.py` is the only Gemini-aware module.
- `tests/test_discovery.py::EXPECTED` asserts the exact registered tool set — update it when adding or removing a tool.

## Testing

- **Async tests call `asyncio.run(...)` inside sync test functions.** `pytest-asyncio` is not a dependency; do not add `@pytest.mark.asyncio`.
- There is no `conftest.py`. Shared doubles live in `tests/fakes.py`, imported as `from tests.fakes import ...`.
- Use `make_tool_context(tmp_path, *, now=None, **settings_overrides)` from `tests/fakes.py` for tool tests — it builds a temp vault with fake memory and constructs `Settings(..., _env_file=None)` so the developer's real `.env` never leaks into tests. Construct `Settings` the same way in any other test that touches config.

## Config

- `Settings` (`config.py`) is pydantic-settings with `env_prefix="JARVIS_"`, read from `.env`. `get_settings()` is `@lru_cache`d for the process.
- Only `JARVIS_VAULT_PATH` is required, and the directory must already exist.
- `LLM_API_KEY` is **unprefixed** (`AliasChoices("LLM_API_KEY", "JARVIS_LLM_API_KEY")`) and typed `SecretStr` so it stays out of logs.
- Never read or echo `.env` — a real one with live credentials exists in the working tree.
- When adding a setting, also add it to `.env.example` and README's env-var table. Several Phase-5 settings (`JARVIS_ENABLED_TOOLS`, `JARVIS_CONFIRM_SIDE_EFFECTS`, `JARVIS_FILE_SANDBOX_ROOT`, ...) are missing from `.env.example` — don't copy that pattern.

## Gotchas

- `.env`, `JARVIS_CHROMA_PATH` (`.jarvis/chroma`) and `reminders_path` are all **CWD-relative** — run Jarvis from the repo root.
- Changing `JARVIS_EMBED_MODEL`, the RAG chunk settings, or the vault path rebuilds the entire memory index.
- Windows-first project: `webrtcvad-wheels` (not `webrtcvad`) is used to avoid needing MSVC; `pyttsx3` goes through SAPI5. Give PowerShell equivalents in docs alongside POSIX ones.
- First run downloads models (fastembed ~70 MB on `--reindex`, faster-whisper ~145 MB on `--voice`); tests never do.
- After a wake-word trigger the detector must be reset **and fed silence** to flush its feature window, or openWakeWord re-fires immediately. Don't "simplify" that away.
- Log events are dotted names via structlog (`agent.tool_call`, `tools.discovered`); `Agent` truncates logged values at 200 chars.

## Repo conventions

- Work lands directly on `main`. Commit subjects follow `Phase N: <imperative summary>` (e.g. `Phase 5: Implement tool discovery and registry`).
- The codebase is built in numbered phases and unbuilt work is marked `TODO(phase-N)` in code. **Do not implement work belonging to a later phase** even when it looks trivial — flag it and move on.
- Run `uv run pytest` and report the result before calling a code change done.
