# Writing a tool

A new capability for Jarvis is **one new file** in `src/jarvis/tools/`. It
needs no registry edits, no agent changes and no CLI wiring. At startup,
`ToolRegistry.discover()` imports every module in that package and registers
each concrete `Tool` subclass it finds.

## Minimal template

Save this as `src/jarvis/tools/dice_tools.py`:

```python
"""Dice rolling."""

import random
from typing import Any

from pydantic import BaseModel, Field

from jarvis.tools.base import Tool


class RollDiceArgs(BaseModel):
    sides: int = Field(default=6, ge=2, le=1000, description="Number of sides on the die.")


class RollDiceTool(Tool):
    name = "roll_dice"                      # unique; what the LLM calls
    description = "Roll a die and return the number."  # tell the LLM when to use it
    args_model = RollDiceArgs               # pydantic model -> JSON schema for the LLM
    requires_confirmation = False           # True if it changes anything in the world
    category = "fun"                        # for grouping in --list-tools and logs

    async def run(self, **kwargs: Any) -> str:
        args = RollDiceArgs.model_validate(kwargs)
        return f"Rolled a d{args.sides}: {random.randint(1, args.sides)}"
```

Then check that it's picked up:

```sh
uv run python -m jarvis.cli --list-tools
```

`roll_dice` appears in the list, and the agent can use it straight away in text,
`--voice` and `--wake` modes.

## Needing dependencies (vault, memory, settings, clock)

Discovery builds each tool with `from_context(context)`, which by default calls
the constructor with no arguments. If your tool needs something, override it
and take what you need from `ToolContext` (`src/jarvis/tools/context.py`):

```python
from typing import Self

from jarvis.tools.context import ToolContext


class CountNotesTool(Tool):
    ...
    def __init__(self, vault: Vault) -> None:
        self._vault = vault

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        return cls(context.vault)
```

Available: `context.settings`, `context.vault`, `context.memory` (embedder,
store, indexer), `context.clock`, `context.file_sandbox_root` and
`context.reminders_path`. If your tool needs something new, add a field to
`ToolContext`.

## Rules

| Rule | Why |
|---|---|
| **Side effect means `requires_confirmation = True`.** This covers anything that writes, sends, buys, deletes or changes state outside Jarvis. | The agent then shows the user the tool, its category and its arguments, and asks `y/N` before every call. |
| **Touching the filesystem means using the sandbox guard.** Resolve the path, then require `is_relative_to(root)`, as `file_tools.resolve_in_sandbox` and `Vault._safe_path` do. | A path from the model must never escape the allowed folder. |
| **Unique `name`.** | Discovery refuses to start if two tools share a name. |
| **Blocking work goes in `await asyncio.to_thread(...)`.** | This keeps the voice loop responsive. |
| **Return short text; don't raise for expected failures** such as no results or network down. | The result goes back to the LLM. Unexpected exceptions are caught by the agent and reported as errors. |
| **Keep third-party imports inside your module**, ideally imported lazily inside functions. | Startup stays fast, and vendor types don't leak. |

## Hiding or disabling tools

- **Prefix the file or class name with `_`** (for example `_experimental.py`) and discovery skips it.
- **Abstract base classes** are skipped automatically.
- **`JARVIS_ENABLED_TOOLS=web_search,get_datetime`** in `.env` limits Jarvis to the listed tools.
- **Problems:** a module that fails to import, or a tool that can't be built, is skipped and reported under "SKIPPED" in `--list-tools`. It doesn't crash Jarvis.

## Testing a tool

Tools are plain async classes, so you can test them without the LLM:

```python
import asyncio
from jarvis.tools.dice_tools import RollDiceTool

def test_roll() -> None:
    assert asyncio.run(RollDiceTool().run(sides=6)).startswith("Rolled a d6")
```

To see the whole loop (the model calling your tool, then answering), script
`tests/fakes.FakeLLMClient` with a `ToolCall` to your tool followed by a text
reply. `tests/test_memory_tool.py` shows the pattern.
