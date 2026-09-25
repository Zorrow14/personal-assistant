"""Test doubles that implement Jarvis interfaces without any network access."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from jarvis.core.interfaces import LLMClient, LLMResponse, Message, ToolSpec
from jarvis.tools.base import Tool


class FakeLLMClient(LLMClient):
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools) if tools is not None else None))
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class EchoArgs(BaseModel):
    text: str


class RecordingTool(Tool):
    """Echoes its input and records each run."""

    name = "echo"
    description = "Echo the text back."
    args_model = EchoArgs

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> str:
        self.runs.append(kwargs)
        return f"echo: {kwargs['text']}"


class DangerousTool(RecordingTool):
    """Same as RecordingTool but gated behind user confirmation."""

    name = "delete_everything"
    description = "Pretend to delete everything."
    requires_confirmation = True


class ExplodingTool(RecordingTool):
    name = "explode"
    description = "Always fails."

    async def run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")
