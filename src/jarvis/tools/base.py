"""Tool contract, registry, and plugin discovery.

Adding a capability = dropping one module into `jarvis/tools/` that defines a
concrete `Tool` subclass. `ToolRegistry.discover()` imports every module in the
package and registers each tool it finds; nothing else needs editing. See
`docs/writing-a-tool.md`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel

from jarvis.core.interfaces import ToolSpec
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from jarvis.tools.context import ToolContext

TOOLS_PACKAGE = "jarvis.tools"

log = get_logger(__name__)


class Tool(ABC):
    """A single action the agent can take.

    Subclasses set `name`, `description` and `args_model` (a pydantic model
    describing the tool's input) and implement `run`.

    `requires_confirmation` is the single source of truth for "this has a
    real-world side effect": set it to True and the agent asks the user y/N
    before every call. `safe` is simply its inverse.

    Tools that need runtime dependencies (the vault, the memory index, a clock,
    settings) override `from_context`; discovery calls it to build the tool.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    requires_confirmation: ClassVar[bool] = False
    category: ClassVar[str] = "general"

    @property
    def safe(self) -> bool:
        """True for tools without side effects (no confirmation needed)."""
        return not self.requires_confirmation

    @classmethod
    def from_context(cls, context: ToolContext) -> Self:
        """Build an instance for auto-discovery.

        The default calls the no-argument constructor. Override this to pull
        dependencies from `context` (e.g. `cls(context.vault)`).
        """
        return cls()

    def schema(self) -> ToolSpec:
        """Describe this tool for the LLM, with a JSON Schema from `args_model`."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )

    def parse_args(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Validate raw LLM-supplied arguments into keyword arguments for `run`.

        Raises:
            pydantic.ValidationError: If `raw` doesn't satisfy `args_model`.
        """
        parsed = self.args_model.model_validate(raw)
        return {field: getattr(parsed, field) for field in type(parsed).model_fields}

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a short textual result for the LLM."""


class DuplicateToolError(ValueError):
    """Two tools claim the same `name`."""


@dataclass(frozen=True)
class DiscoveryProblem:
    """A module or tool that discovery had to skip, and why."""

    where: str
    error: str


def discover_tool_classes(
    package: str = TOOLS_PACKAGE,
) -> tuple[list[type[Tool]], list[DiscoveryProblem]]:
    """Find every concrete `Tool` subclass defined in `package`'s modules.

    Skips modules and classes whose names start with `_`, abstract classes,
    and classes merely imported into a module (each is found where it's
    defined). A module that fails to import is reported, not fatal.

    Returns:
        (tool classes sorted by name, problems encountered).

    Raises:
        DuplicateToolError: If two classes declare the same tool name.
    """
    pkg = importlib.import_module(package)
    classes: list[type[Tool]] = []
    problems: list[DiscoveryProblem] = []
    for info in pkgutil.iter_modules(pkg.__path__, prefix=f"{package}."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:
            problems.append(DiscoveryProblem(info.name, f"import failed: {type(exc).__name__}: {exc}"))
            log.error("tools.import_failed", module=info.name, error=str(exc))
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not _is_discoverable(obj, module.__name__):
                continue
            missing = [attr for attr in ("name", "description", "args_model") if not hasattr(obj, attr)]
            if missing:
                problems.append(
                    DiscoveryProblem(_qualname(obj), f"missing class attribute(s): {', '.join(missing)}")
                )
                continue
            classes.append(obj)

    by_name: dict[str, type[Tool]] = {}
    for cls in classes:
        other = by_name.setdefault(cls.name, cls)
        if other is not cls:
            raise DuplicateToolError(
                f"tool name {cls.name!r} is defined twice: {_qualname(other)} and {_qualname(cls)}"
            )
    return sorted(classes, key=lambda c: c.name), problems


def _is_discoverable(obj: type, module_name: str) -> bool:
    return (
        issubclass(obj, Tool)
        and obj is not Tool
        and obj.__module__ == module_name
        and not obj.__name__.startswith("_")
        and not inspect.isabstract(obj)
    )


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


class ToolRegistry:
    """The set of tools available to the agent, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.discovery_problems: list[DiscoveryProblem] = []

    def register(self, tool: Tool) -> None:
        """Add `tool`; raises DuplicateToolError if its name is already taken."""
        if tool.name in self._tools:
            raise DuplicateToolError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def discover(self, package: str = TOOLS_PACKAGE, *, context: ToolContext) -> list[str]:
        """Import `package`'s modules and register every tool found, built via `from_context`.

        Idempotent: tools already registered from the same class are skipped.
        A tool whose `from_context` fails is recorded in `discovery_problems`
        and skipped, so one broken plugin can't take Jarvis down.

        Returns:
            Names of the tools newly registered.

        Raises:
            DuplicateToolError: If two tools share a name.
        """
        classes, problems = discover_tool_classes(package)
        self.discovery_problems.extend(problems)
        added: list[str] = []
        for cls in classes:
            existing = self._tools.get(cls.name)
            if existing is not None:
                if type(existing) is cls:
                    continue
                raise DuplicateToolError(
                    f"tool name {cls.name!r} is already registered by {_qualname(type(existing))}; "
                    f"cannot also register {_qualname(cls)}"
                )
            try:
                tool = cls.from_context(context)
            except Exception as exc:
                hint = " (override from_context to supply constructor arguments)" if isinstance(exc, TypeError) else ""
                self.discovery_problems.append(
                    DiscoveryProblem(_qualname(cls), f"could not be built: {type(exc).__name__}: {exc}{hint}")
                )
                log.error("tools.build_failed", tool=cls.name, error=str(exc))
                continue
            self._tools[cls.name] = tool
            added.append(cls.name)
        log.info(
            "tools.discovered",
            tools={name: self._tools[name].category for name in added},
            problems=len(problems),
        )
        return added

    def restrict(self, allowed: Iterable[str]) -> list[str]:
        """Keep only tools named in `allowed`; return any allowed names that don't exist."""
        wanted = set(allowed)
        for name in list(self._tools):
            if name not in wanted:
                del self._tools[name]
        return sorted(wanted - set(self._tools))

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Names of all registered tools, in registration order."""
        return list(self._tools)

    def schemas(self) -> list[ToolSpec]:
        """Schemas for every registered tool, for passing to the LLM."""
        return [tool.schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
