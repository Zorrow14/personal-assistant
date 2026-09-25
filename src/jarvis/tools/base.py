"""Tool contract and registry."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel

from jarvis.core.interfaces import ToolSpec


class Tool(ABC):
    """A single action the agent can take.

    Subclasses set `name`, `description` and `args_model` (a pydantic model
    describing the tool's input) and implement `run`. Set
    `requires_confirmation = True` for anything risky; the agent then asks the
    user before running it.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    requires_confirmation: ClassVar[bool] = False

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


class ToolRegistry:
    """The set of tools available to the agent, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add `tool`; raises ValueError if its name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

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

    # TODO(phase-5): plugin auto-discovery.
