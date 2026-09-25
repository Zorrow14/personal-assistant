---
name: add-tool
description: Add a new tool (capability) to Jarvis — scaffolds the Tool subclass, wires dependencies through ToolContext, sets the confirmation and sandbox guards, writes the test, and verifies discovery. Use when asked to give Jarvis a new ability, add a tool, or expose a new action to the agent.
---

# Adding a Jarvis tool

Read `@docs/writing-a-tool.md` first — it is the authoritative contract. This skill is the ordered workflow around it.

Tool argument (if given): `$ARGUMENTS` — the capability to add.

## 1. Decide the shape before writing code

Answer these, and ask the user only if the request is genuinely ambiguous:

- **Module name.** One new file at `src/jarvis/tools/<area>_tools.py`, or add a class to an existing area module if it clearly belongs (`time_tools.py`, `file_tools.py`, `web_tools.py`, `vault_tools.py`, `memory_tools.py`). A leading `_` on the file or class hides it from discovery — only use that deliberately.
- **`name`** — unique across every registered tool; a duplicate is a fatal `DuplicateToolError`. Check `uv run python -m jarvis.cli --list-tools` for what exists.
- **`category`** — groups the tool in `--list-tools` and in the confirmation banner.
- **Does it have a real-world side effect?** Writing a file, sending anything, mutating state outside the process ⇒ `requires_confirmation = True`. This ClassVar is the *only* signal the agent's confirmation gate reads. Reads are `False`.
- **Dependencies.** Needs settings, the vault, memory, or the clock? Override `from_context`. If it needs a *kind* of dependency `ToolContext` doesn't carry yet, add a field to `ToolContext` in `src/jarvis/tools/context.py` (frozen dataclass) and pass it from `cli.py`.

## 2. Write the tool

```python
from typing import ClassVar, Self

from pydantic import BaseModel, Field

from jarvis.tools.base import Tool
from jarvis.tools.context import ToolContext


class MyToolArgs(BaseModel):
    query: str = Field(description="What the model should pass here.")


class MyTool(Tool):
    name: ClassVar[str] = "my_tool"
    description: ClassVar[str] = "One line the LLM reads to decide when to call this."
    args_model: ClassVar[type[BaseModel]] = MyToolArgs
    category: ClassVar[str] = "general"
    requires_confirmation: ClassVar[bool] = False

    def __init__(self, *, dep: SomeDep) -> None:
        self._dep = dep

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        return cls(dep=context.settings)

    async def run(self, **kwargs) -> str:
        args = self.parse_args(kwargs)
        ...
        return "short human-readable result"
```

Rules to hold to:

- `description` and every `Field(description=...)` are prompt surface — the LLM picks the tool from them. Write them for the model, not for a human reader.
- Blocking work (network, disk, model inference) goes in `await asyncio.to_thread(...)`.
- Filesystem paths must pass the sandbox guard: resolve, then `is_relative_to(root)`. Reuse `file_tools.resolve_in_sandbox` or `Vault._safe_path` rather than re-rolling it. The root is `context.file_sandbox_root`.
- Return short text for expected failures ("no results found") instead of raising. Unexpected exceptions are caught by the agent and fed back to the model as errors, so don't add your own try/except theater.
- Keep third-party imports inside the module, lazily if the dependency is heavy — a failed import turns into a non-fatal `DiscoveryProblem`, which silently drops the tool.

## 3. Register nothing, but update the expected set

Discovery is automatic. However `tests/test_discovery.py::EXPECTED` pins the exact registered tool set — add the new `name` there or that test fails.

## 4. Test it

New file `tests/test_<area>_tools.py`, or extend the existing one. Conventions in this repo:

- Async tests are **sync functions calling `asyncio.run(...)`**. `pytest-asyncio` is not installed; never add `@pytest.mark.asyncio`.
- Build the tool via `make_tool_context` from `tests/fakes.py`:

  ```python
  from tests.fakes import make_tool_context

  def test_my_tool(tmp_path):
      context = make_tool_context(tmp_path)
      tool = MyTool.from_context(context)
      result = asyncio.run(tool.run(query="hi"))
      assert "..." in result
  ```

- Pass `now=` to `make_tool_context` to freeze the clock; pass settings overrides as keyword args.
- Cover: the happy path, a validation rejection via `parse_args`, and — for anything touching the filesystem — a path-escape attempt that must be refused.
- If `requires_confirmation = True`, assert it, and check `tests/test_confirmation.py` for the gate's existing coverage.

## 5. Verify

```sh
uv run python -m jarvis.cli --list-tools   # new tool listed, nothing under SKIPPED
uv run pytest                              # full offline suite
```

Report both results. If the tool appears under SKIPPED, the reason printed there is the import or build failure to fix.

## 6. Documentation

If the tool adds a setting, add it to `.env.example` and README's env-var table in the same change.
