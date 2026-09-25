"""Gemini translation and retry tests. A fake SDK object stands in for the network."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from jarvis.config import Settings
from jarvis.core.interfaces import LLMError, Message, ToolCall, ToolSpec
from jarvis.llm.gemini_client import (
    GeminiClient,
    from_gemini_response,
    to_gemini_contents,
)


def _response(parts: list[dict[str, Any]], finish: str = "STOP") -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(
        {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]}
    )


def _api_error(code: int, status: str) -> genai_errors.APIError:
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": "nope", "status": status}}
    )


class FakeModels:
    """Mimics `client.aio.models`: raises queued errors, then returns the response."""

    def __init__(self, response: types.GenerateContentResponse, errors: list[Exception]) -> None:
        self.response = response
        self.errors = list(errors)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.response


def _client(
    response: types.GenerateContentResponse | None = None, errors: list[Exception] | None = None
) -> tuple[GeminiClient, FakeModels, list[float]]:
    models = FakeModels(response or _response([{"text": "hi"}]), errors or [])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = GeminiClient(
        api_key="test",
        model="gemini-test",
        max_tokens=123,
        temperature=0.5,
        sdk_client=SimpleNamespace(aio=SimpleNamespace(models=models)),  # type: ignore[arg-type]
        sleep=fake_sleep,
    )
    return client, models, sleeps


# --- request translation -------------------------------------------------------


def test_messages_map_to_gemini_roles_and_parts() -> None:
    contents = to_gemini_contents(
        [
            Message(role="user", content="log it"),
            Message(
                role="assistant",
                content="On it.",
                tool_calls=[ToolCall(id="jarvis-call-abc", name="write_task_note", input={"a": 1})],
            ),
            Message(role="tool", content="Saved", tool_call_id="jarvis-call-abc", tool_name="write_task_note"),
        ]
    )

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[0].parts[0].text == "log it"
    model_parts = contents[1].parts
    assert model_parts[0].text == "On it."
    assert model_parts[1].function_call.name == "write_task_note"
    assert model_parts[1].function_call.args == {"a": 1}
    assert model_parts[1].function_call.id is None  # synthetic ids are never sent
    fr = contents[2].parts[0].function_response
    assert (fr.name, fr.response, fr.id) == ("write_task_note", {"output": "Saved"}, None)


def test_parallel_tool_results_merge_into_one_turn_and_errors_are_flagged() -> None:
    contents = to_gemini_contents(
        [
            Message(role="tool", content="ok", tool_call_id="real-1", tool_name="a"),
            Message(role="tool", content="boom", tool_call_id="real-2", tool_name="b", is_error=True),
        ]
    )

    assert len(contents) == 1
    responses = [p.function_response for p in contents[0].parts]
    assert [(r.id, r.response) for r in responses] == [
        ("real-1", {"output": "ok"}),
        ("real-2", {"error": "boom"}),
    ]


def test_assistant_turn_replays_raw_content_with_thought_signature() -> None:
    raw = _response(
        [{"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "c2ln"}]
    )
    contents = to_gemini_contents(
        [Message(role="assistant", tool_calls=[ToolCall("x", "f", {})], raw=raw)]
    )

    assert contents[0].role == "model"
    assert contents[0].parts[0].thought_signature == b"sig"


def test_request_carries_config_and_function_declarations() -> None:
    client, models, _ = _client()
    spec = ToolSpec("write_task_note", "Write a note.", {"type": "object", "properties": {}})

    asyncio.run(client.complete([Message(role="user", content="hi")], tools=[spec]))

    [call] = models.calls
    assert call["model"] == "gemini-test"
    config: types.GenerateContentConfig = call["config"]
    assert config.max_output_tokens == 123
    assert config.temperature == 0.5
    assert "Jarvis" in str(config.system_instruction)
    [decl] = config.tools[0].function_declarations
    assert decl.name == "write_task_note"
    assert decl.parameters_json_schema == spec.input_schema


# --- response translation ------------------------------------------------------


def test_function_call_response_becomes_tool_calls() -> None:
    raw = _response(
        [
            {"text": "thinking...", "thought": True},
            {"functionCall": {"name": "write_task_note", "args": {"command": "x"}}},
        ]
    )

    result = from_gemini_response(raw)

    assert result.stop_reason == "tool_use"
    assert result.text is None  # thought parts are not user-facing text
    [call] = result.tool_calls
    assert call.name == "write_task_note"
    assert call.input == {"command": "x"}
    assert call.id.startswith("jarvis-call-")
    assert result.raw is raw


@pytest.mark.parametrize(
    ("finish", "expected"),
    [("STOP", "end_turn"), ("MAX_TOKENS", "max_tokens"), ("SAFETY", "safety"), ("OTHER", "other")],
)
def test_finish_reasons_map_to_neutral_stop_reasons(finish: str, expected: str) -> None:
    result = from_gemini_response(_response([{"text": " Done. "}], finish=finish))
    assert result.text == "Done."
    assert result.stop_reason == expected


def test_blocked_prompt_has_no_candidates() -> None:
    raw = types.GenerateContentResponse.model_validate({"promptFeedback": {"blockReason": "SAFETY"}})
    result = from_gemini_response(raw)
    assert (result.text, result.tool_calls, result.stop_reason) == (None, [], "safety")


# --- retries -------------------------------------------------------------------


def test_retries_rate_limits_with_exponential_backoff() -> None:
    client, models, sleeps = _client(
        errors=[_api_error(429, "RESOURCE_EXHAUSTED"), _api_error(503, "UNAVAILABLE")]
    )

    result = asyncio.run(client.complete([Message(role="user", content="hi")]))

    assert result.text == "hi"
    assert len(models.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_attempts() -> None:
    client, models, sleeps = _client(errors=[_api_error(429, "RESOURCE_EXHAUSTED")] * 5)

    with pytest.raises(LLMError, match="429 RESOURCE_EXHAUSTED"):
        asyncio.run(client.complete([Message(role="user", content="hi")]))

    assert len(models.calls) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_non_retryable_error_fails_immediately() -> None:
    client, models, sleeps = _client(errors=[_api_error(400, "INVALID_ARGUMENT")])

    with pytest.raises(LLMError, match="400"):
        asyncio.run(client.complete([Message(role="user", content="hi")]))

    assert len(models.calls) == 1
    assert sleeps == []


def test_from_settings_requires_api_key(tmp_path: Path) -> None:
    settings = Settings(vault_path=tmp_path, _env_file=None)  # type: ignore[call-arg]
    if settings.llm_api_key is not None:
        pytest.skip("LLM_API_KEY set in the environment")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        GeminiClient.from_settings(settings)
