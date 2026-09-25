"""The agent: turns a natural-language command into actions via tool-calling.

Everything here is expressed in the neutral types from `core.interfaces`; the
agent never knows which LLM provider it is talking to.
"""

import json
from collections.abc import Callable

from pydantic import ValidationError

from jarvis.core.interfaces import LLMClient, LLMResponse, Message, ToolCall
from jarvis.logging import get_logger
from jarvis.tools.base import ToolRegistry

DECLINED_RESULT = "User declined this action."
MAX_ITERATIONS_REPLY = "Stopped: hit the max tool-iteration limit."
_LOG_PREVIEW_CHARS = 200

ConfirmFn = Callable[[ToolCall], bool]

log = get_logger(__name__)


def confirm_on_cli(call: ToolCall) -> bool:
    """Show a pending tool call and ask the user y/N on the terminal."""
    args = json.dumps(call.input, indent=2, ensure_ascii=False)
    print(f"\nJarvis wants to run '{call.name}' with:\n{args}")
    try:
        answer = input("Allow? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


class Agent:
    """Runs the LLM tool-calling loop and keeps conversation history.

    For each user input: ask the LLM (offering every registered tool); if it
    requests tools, run them, feed the results back, and ask again; stop when
    it answers in plain text or after `max_iterations` LLM calls.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_iterations: int = 8,
        confirm: ConfirmFn | None = None,
    ) -> None:
        """
        Args:
            llm: Any `LLMClient` implementation.
            registry: Tools the LLM may call.
            max_iterations: Cap on LLM calls per `run`.
            confirm: Approval callback for tools with `requires_confirmation`;
                defaults to asking on the terminal.
        """
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self.history: list[Message] = []
        self._confirm_fn = confirm

    async def run(self, user_input: str) -> str:
        """Handle one user input end to end and return the reply text.

        If the LLM fails mid-turn, the turn is removed from history (tools that
        already ran keep their effects) and the error propagates.
        """
        checkpoint = len(self.history)
        self.history.append(Message(role="user", content=user_input))
        try:
            return await self._loop()
        except BaseException:
            del self.history[checkpoint:]
            raise

    def reset(self) -> None:
        """Forget the conversation so far."""
        self.history.clear()

    async def _loop(self) -> str:
        # TODO(phase-4): trim/summarise long histories before they hit the context limit.
        tools = self.registry.schemas() or None
        for _ in range(self.max_iterations):
            response = await self.llm.complete(tuple(self.history), tools=tools)
            self.history.append(
                Message(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=list(response.tool_calls),
                    raw=response.raw,
                )
            )
            if not response.tool_calls:
                return response.text or _empty_reply(response)
            for call in response.tool_calls:
                result, is_error = await self._execute(call)
                self.history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=is_error,
                    )
                )
        log.warning("agent.max_iterations", limit=self.max_iterations)
        return MAX_ITERATIONS_REPLY

    async def _execute(self, call: ToolCall) -> tuple[str, bool]:
        """Run one tool call; never raises. Returns (result text, is_error)."""
        log.info("agent.tool_call", tool=call.name, call_id=call.id, input=_preview(call.input))
        tool = self.registry.get(call.name)
        if tool is None:
            available = ", ".join(self.registry.names()) or "none"
            result, is_error = f"Error: unknown tool {call.name!r}. Available tools: {available}.", True
        elif tool.requires_confirmation and not self._confirm(call):
            result, is_error = DECLINED_RESULT, False
        else:
            try:
                result, is_error = await tool.run(**tool.parse_args(call.input)), False
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}" for err in exc.errors()
                )
                result, is_error = f"Error: invalid arguments for {call.name}: {problems}", True
            except Exception as exc:
                log.exception("agent.tool_failed", tool=call.name)
                result, is_error = f"Error: {call.name} failed: {type(exc).__name__}: {exc}", True
        log.info("agent.tool_result", tool=call.name, is_error=is_error, result=_preview(result))
        return result, is_error

    def _confirm(self, call: ToolCall) -> bool:
        """Safety gate for risky tools: ask the user before running `call`."""
        # Looked up at call time so tests can monkeypatch `confirm_on_cli`.
        return (self._confirm_fn or confirm_on_cli)(call)


def _empty_reply(response: LLMResponse) -> str:
    return f"(No reply from the model; stop reason: {response.stop_reason}.)"


def _preview(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= _LOG_PREVIEW_CHARS else text[:_LOG_PREVIEW_CHARS] + "..."
