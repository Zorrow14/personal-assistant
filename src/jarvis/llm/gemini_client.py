"""Google Gemini implementation of `LLMClient`, using the google-genai SDK.

This is the only module that knows about Gemini. It maps Jarvis's neutral
types to Gemini's wire format and back:

- roles: user -> "user", assistant -> "model", tool -> "user" carrying a
  `functionResponse` part (consecutive same-role turns are merged)
- tools: `ToolSpec` -> `FunctionDeclaration` (JSON Schema passed through)
- replies: text parts + `functionCall` parts + finish reason -> `LLMResponse`
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from jarvis.config import Settings
from jarvis.core.interfaces import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from jarvis.logging import get_logger

SYSTEM_PROMPT = """\
You are Jarvis, a personal assistant running on the user's computer.
- Be concise: reply in one or two short sentences unless asked for more.
- Act through your tools instead of describing what you would do.
- When the user asks you to log, record or note something, or once you have
  completed a task for them, record it in their vault with the note-writing tool.
- Never claim a tool succeeded unless its result says so; report errors plainly.
"""

RETRYABLE_STATUS_CODES = frozenset({429, 500, 503})
"""429 RESOURCE_EXHAUSTED (free-tier rate limit) plus transient server errors."""
MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 1.0

# Gemini often omits function-call ids. We mint our own so the agent can link
# results to calls, and strip them again before sending back.
_SYNTHETIC_ID_PREFIX = "jarvis-call-"

_FINISH_REASONS: dict[types.FinishReason, StopReason] = {
    types.FinishReason.STOP: "end_turn",
    types.FinishReason.MAX_TOKENS: "max_tokens",
    types.FinishReason.SAFETY: "safety",
    types.FinishReason.RECITATION: "safety",
    types.FinishReason.BLOCKLIST: "safety",
    types.FinishReason.PROHIBITED_CONTENT: "safety",
    types.FinishReason.SPII: "safety",
    types.FinishReason.MALFORMED_FUNCTION_CALL: "error",
    types.FinishReason.UNEXPECTED_TOOL_CALL: "error",
    types.FinishReason.TOO_MANY_TOOL_CALLS: "error",
}

log = get_logger(__name__)


class GeminiClient(LLMClient):
    """`LLMClient` backed by the Gemini API, with backoff on rate limits."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str = SYSTEM_PROMPT,
        sdk_client: genai.Client | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """
        Args:
            api_key: Gemini API key (Google AI Studio).
            model: Model name, e.g. "gemini-2.5-flash".
            max_tokens: Output token cap (Gemini's `max_output_tokens`).
            temperature: Sampling temperature.
            system_prompt: Instruction sent with every request.
            sdk_client: Pre-built SDK client (injectable for tests).
            sleep: Async sleep used between retries (injectable for tests).
        """
        self._client = sdk_client or genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiClient":
        """Build a client from `Settings`.

        Raises:
            LLMError: If no API key is configured.
        """
        if settings.llm_api_key is None:
            raise LLMError(
                "LLM_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and add it to .env"
            )
        return cls(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send the conversation to Gemini and translate its reply."""
        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
            tools=[types.Tool(function_declarations=[to_function_declaration(t) for t in tools])]
            if tools
            else None,
            # We run tools ourselves in the agent loop.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        started = time.perf_counter()
        response = await self._generate_with_retry(to_gemini_contents(messages), config)
        result = from_gemini_response(response)
        usage = result.usage or TokenUsage()
        log.info(
            "llm.complete",
            model=self._model,
            messages=len(messages),
            finish_reason=_finish_reason_name(response),
            tool_calls=len(result.tool_calls),
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return result

    async def aclose(self) -> None:
        """Close the SDK's async HTTP session."""
        await self._client.aio.aclose()

    async def _generate_with_retry(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                if exc.code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS:
                    raise LLMError(
                        f"Gemini request failed ({exc.code} {exc.status}): {exc.message}",
                        status_code=exc.code,
                    ) from exc
                delay = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
                log.warning(
                    "llm.retry", code=exc.code, status=exc.status, attempt=attempt, delay_s=delay
                )
                await self._sleep(delay)
            except Exception as exc:
                raise LLMError(f"Gemini request failed: {type(exc).__name__}: {exc}") from exc
        raise AssertionError("unreachable")


def to_function_declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    """Convert a neutral `ToolSpec` into a Gemini function declaration."""
    return types.FunctionDeclaration(
        name=spec.name,
        description=spec.description,
        parameters_json_schema=spec.input_schema,
    )


def to_gemini_contents(messages: Sequence[Message]) -> list[types.Content]:
    """Convert neutral messages into Gemini `Content`s, merging same-role runs.

    Gemini expects the results of parallel tool calls in a single user turn, so
    consecutive messages mapping to the same role are combined.
    """
    contents: list[types.Content] = []
    for message in messages:
        content = _to_content(message)
        if content is None:
            continue
        if contents and contents[-1].role == content.role:
            previous = contents[-1]
            contents[-1] = types.Content(
                role=previous.role, parts=[*(previous.parts or []), *(content.parts or [])]
            )
        else:
            contents.append(content)
    return contents


def from_gemini_response(response: types.GenerateContentResponse) -> LLMResponse:
    """Convert a Gemini response into a neutral `LLMResponse`."""
    usage = usage_from_gemini(response)
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None:
        blocked = response.prompt_feedback is not None and response.prompt_feedback.block_reason
        return LLMResponse(
            text=None, stop_reason="safety" if blocked else "other", raw=response, usage=usage
        )

    parts = candidate.content.parts if candidate.content and candidate.content.parts else []
    text = "".join(p.text for p in parts if p.text and not p.thought).strip() or None
    calls = [
        ToolCall(
            id=fc.id or f"{_SYNTHETIC_ID_PREFIX}{uuid.uuid4().hex[:12]}",
            name=fc.name or "",
            input=dict(fc.args or {}),
        )
        for p in parts
        if (fc := p.function_call) is not None
    ]
    stop_reason: StopReason = "tool_use"
    if not calls:
        finish = candidate.finish_reason
        stop_reason = _FINISH_REASONS.get(finish, "other") if finish else "other"
    return LLMResponse(
        text=text, tool_calls=calls, stop_reason=stop_reason, raw=response, usage=usage
    )


def usage_from_gemini(response: types.GenerateContentResponse) -> TokenUsage | None:
    """Token counts from Gemini's `usage_metadata`, or None if it sent none."""
    meta = response.usage_metadata
    if meta is None:
        return None
    return TokenUsage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        thinking_tokens=meta.thoughts_token_count or 0,
    )


def _to_content(message: Message) -> types.Content | None:
    if message.role == "user":
        return types.Content(role="user", parts=[types.Part(text=message.content)])

    if message.role == "assistant":
        # Replay Gemini's own turn verbatim when we have it: this preserves
        # thought signatures, which the API requires alongside function calls.
        original = _original_model_content(message.raw)
        if original is not None:
            return original
        parts: list[types.Part] = []
        if message.content:
            parts.append(types.Part(text=message.content))
        parts.extend(
            types.Part(
                function_call=types.FunctionCall(
                    id=_real_call_id(call.id), name=call.name, args=call.input
                )
            )
            for call in message.tool_calls
        )
        return types.Content(role="model", parts=parts) if parts else None

    # role == "tool"
    key = "error" if message.is_error else "output"
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=_real_call_id(message.tool_call_id),
                    name=message.tool_name or "unknown",
                    response={key: message.content},
                )
            )
        ],
    )


def _original_model_content(raw: object) -> types.Content | None:
    if not isinstance(raw, types.GenerateContentResponse) or not raw.candidates:
        return None
    content = raw.candidates[0].content
    if content is None or not content.parts:
        return None
    return types.Content(role="model", parts=content.parts)


def _real_call_id(call_id: str | None) -> str | None:
    if call_id is None or call_id.startswith(_SYNTHETIC_ID_PREFIX):
        return None
    return call_id


def _finish_reason_name(response: types.GenerateContentResponse) -> str | None:
    if not response.candidates or response.candidates[0].finish_reason is None:
        return None
    return response.candidates[0].finish_reason.name
