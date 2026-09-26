"""The agent: turns a natural-language command into actions via tool-calling.

Everything here is expressed in the neutral types from `core.interfaces`; the
agent never knows which LLM provider it is talking to.
"""

import asyncio
import json
import textwrap
from collections.abc import Callable

from pydantic import ValidationError

from jarvis.core.events import (
    TOOL,
    TOOL_COMPLETED,
    TOOL_DECLINED,
    TOOL_ERROR,
    TOOL_STARTED,
    EventBus,
)
from jarvis.core.interfaces import LLMClient, LLMResponse, Message, ToolCall
from jarvis.logging import get_logger
from jarvis.tools.base import ToolRegistry

DECLINED_RESULT = "User declined this action."
MAX_ITERATIONS_REPLY = "Stopped: hit the max tool-iteration limit."
_LOG_PREVIEW_CHARS = 200

ConfirmFn = Callable[[ToolCall], bool]

log = get_logger(__name__)


def confirm_on_cli(call: ToolCall, *, category: str = "general") -> bool:
    """Show a pending side-effect action unmistakably and ask the user y/N on the terminal.

    Anything but an explicit "y"/"yes" (including Enter, EOF) declines.
    """
    args = json.dumps(call.input, indent=2, ensure_ascii=False)
    bar = "=" * 64
    print(
        f"\n{bar}\n"
        f"  CONFIRM: Jarvis wants to run an action with side effects\n"
        f"    tool:      {call.name}\n"
        f"    category:  {category}\n"
        f"    arguments:\n{textwrap.indent(args, '      ')}\n"
        f"{bar}"
    )
    try:
        answer = input("Allow this action? [y/N] ")
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
        confirm_side_effects: bool = True,
        events: EventBus | None = None,
    ) -> None:
        """
        Args:
            llm: Any `LLMClient` implementation.
            registry: Tools the LLM may call.
            max_iterations: Cap on LLM calls per `run`.
            confirm: Approval callback for tools with `requires_confirmation`;
                defaults to asking on the terminal.
            confirm_side_effects: Master switch for the gate. False runs
                side-effect tools without asking (each bypass is logged).
            events: Where to publish `tool` events (e.g. for the web panel);
                None publishes nothing.
        """
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations
        self.history: list[Message] = []
        self._confirm_fn = confirm
        self.confirm_side_effects = confirm_side_effects
        self.events = events
        # One turn at a time: the panel and the wake-word loop can both call `run`.
        self._turn_lock = asyncio.Lock()

    async def run(self, user_input: str) -> str:
        """Handle one user input end to end and return the reply text.

        Concurrent calls are queued and run one after another, so turns never
        interleave in the history. If the LLM fails mid-turn, the turn is
        removed from history (tools that already ran keep their effects) and
        the error propagates.
        """
        async with self._turn_lock:
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
        category = tool.category if tool else "unknown"
        if tool is None:
            available = ", ".join(self.registry.names()) or "none"
            result, is_error = (
                f"Error: unknown tool {call.name!r}. Available tools: {available}.",
                True,
            )
            status = TOOL_ERROR
        elif tool.requires_confirmation and not self._confirm(call):
            result, is_error = DECLINED_RESULT, False
            status = TOOL_DECLINED
        else:
            self._publish_tool(call, category, TOOL_STARTED)
            try:
                result, is_error = await tool.run(**tool.parse_args(call.input)), False
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}"
                    for err in exc.errors()
                )
                result, is_error = f"Error: invalid arguments for {call.name}: {problems}", True
            except Exception as exc:
                log.exception("agent.tool_failed", tool=call.name)
                result, is_error = f"Error: {call.name} failed: {type(exc).__name__}: {exc}", True
            status = TOOL_ERROR if is_error else TOOL_COMPLETED
        log.info("agent.tool_result", tool=call.name, is_error=is_error, result=_preview(result))
        self._publish_tool(call, category, status, result)
        return result, is_error

    def _publish_tool(
        self, call: ToolCall, category: str, status: str, result: str | None = None
    ) -> None:
        """Tell observers about a tool call; a no-op without an event bus."""
        if self.events is None:
            return
        data = {
            "id": call.id,
            "name": call.name,
            "category": category,
            "args": dict(call.input),
            "status": status,
        }
        if result is not None:
            data["result"] = _preview(result)
        self.events.emit(TOOL, **data)

    def _confirm(self, call: ToolCall) -> bool:
        """Safety gate for risky tools: ask the user before running `call`."""
        if not self.confirm_side_effects:
            log.warning(
                "agent.confirmation_bypassed", tool=call.name, reason="confirm_side_effects=False"
            )
            return True
        if self._confirm_fn is not None:
            return self._confirm_fn(call)
        tool = self.registry.get(call.name)
        # Looked up at call time so tests can monkeypatch `confirm_on_cli`.
        return confirm_on_cli(call, category=tool.category if tool else "general")


def _empty_reply(response: LLMResponse) -> str:
    return f"(No reply from the model; stop reason: {response.stop_reason}.)"


def _preview(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= _LOG_PREVIEW_CHARS else text[:_LOG_PREVIEW_CHARS] + "..."
