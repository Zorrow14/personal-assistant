"""Local panel tests: Starlette's TestClient and fakes. No browser, no real socket, no mic."""

import asyncio
import socket
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from jarvis.cli import parse_args
from jarvis.core.agent import Agent
from jarvis.core.events import (
    ERROR,
    HELLO,
    NOTICE,
    REPLY,
    STATE,
    TOOL,
    TRANSCRIPT,
    Event,
    EventBus,
)
from jarvis.core.interfaces import (
    AudioSamples,
    FrameSource,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
    WakeWordDetector,
)
from jarvis.core.voice_loop import WakeWordLoop
from jarvis.obs.metrics import MetricsRecorder
from jarvis.server import app as app_module
from jarvis.server.app import (
    INDEX_HTML,
    NonLoopbackHostError,
    bind_loopback_socket,
    check_loopback,
    create_app,
    host_allowed,
    origin_allowed,
    panel_url,
)
from jarvis.server.controller import (
    BUSY_MESSAGE,
    EXIT_HINT,
    MAX_TEXT_CHARS,
    HandsFreeVoice,
    ManualWakeTrigger,
    PanelController,
    PushToTalkVoice,
)
from jarvis.tools.base import ToolRegistry
from tests.fakes import (
    FakeFrameSource,
    FakeLLMClient,
    FakeSTTEngine,
    FakeTTSEngine,
    FakeVAD,
    RecordingTool,
)

BASE = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"
PANEL_ORIGIN = {"origin": BASE}


# --- helpers -----------------------------------------------------------------


def _agent(bus: EventBus, responses: Sequence[LLMResponse], *tools: Any) -> Agent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return Agent(FakeLLMClient(responses), registry, events=bus)


def _llm(agent: Agent) -> FakeLLMClient:
    assert isinstance(agent.llm, FakeLLMClient)
    return agent.llm


def _client(bus: EventBus, controller: PanelController, base_url: str = BASE) -> TestClient:
    return TestClient(create_app(bus, controller, port=8000), base_url=base_url)


def _receive_until(
    ws: Any, done: Callable[[dict[str, Any]], bool], limit: int = 40
) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    for _ in range(limit):
        event = ws.receive_json()
        seen.append(event)
        if done(event):
            return seen
    raise AssertionError(f"expected event not received; got {seen}")


def _summary(events: Sequence[dict[str, Any]]) -> list[tuple[str, Any]]:
    key = {STATE: "state", TRANSCRIPT: "text", REPLY: "text", TOOL: "status"}
    return [(e["type"], e["data"].get(key.get(e["type"], ""), "")) for e in events]


def _is_reply(event: dict[str, Any]) -> bool:
    return event["type"] == REPLY


def _is_idle(event: dict[str, Any]) -> bool:
    return event["type"] == STATE and event["data"]["state"] == "idle"


class SlowSource(FakeFrameSource):
    """Endless silence, a couple of milliseconds per read like a real mic."""

    def read(self, n_samples: int) -> AudioSamples:
        time.sleep(0.002)
        return super().read(n_samples)


class FakeMic(SlowSource):
    """A switchable mic: counts opens and closes."""

    def __init__(self) -> None:
        super().__init__()
        self.starts = self.closes = 0

    def start(self) -> None:
        self.starts += 1

    def close(self) -> None:
        self.closes += 1


class SilentDetector(WakeWordDetector):
    """Never hears the wake word by itself."""

    def __init__(self) -> None:
        self.resets = 0

    @property
    def frame_samples(self) -> int:
        return 1280

    def process(self, frame: AudioSamples) -> float:
        return 0.0

    def reset(self) -> None:
        self.resets += 1


class EndlessRecorder:
    """Keeps recording until the loop is stopped."""

    def record_command(self, source: FrameSource) -> AudioSamples:
        while True:
            source.read(480)


def _push_to_talk(
    agent: Agent,
    bus: EventBus,
    *,
    transcripts: list[str],
    recorder: Any = None,
) -> tuple[PushToTalkVoice, FakeMic, FakeTTSEngine]:
    mic, tts, stt = FakeMic(), FakeTTSEngine(), FakeSTTEngine(transcripts)

    def make_loop(detector: WakeWordDetector, source: FrameSource) -> WakeWordLoop:
        return WakeWordLoop(
            agent,
            stt,
            tts,
            detector,
            recorder or FakeVAD(),
            source,
            display=lambda _line: None,
            events=bus,
        )

    return PushToTalkVoice(make_loop, mic, bus), mic, tts


async def _wait_for(
    queue: asyncio.Queue[Event], done: Callable[[Event], bool], within: float = 3.0
) -> list[Event]:
    seen: list[Event] = []

    async def scan() -> list[Event]:
        while True:
            event = await queue.get()
            seen.append(event)
            if done(event):
                return seen

    try:
        return await asyncio.wait_for(scan(), within)
    except TimeoutError:
        raise AssertionError(f"expected event not published; got {seen}") from None


def _state(name: str) -> Callable[[Event], bool]:
    return lambda e: e.type == STATE and e.data["state"] == name


# --- HTTP ----------------------------------------------------------------------


def test_health_returns_ok() -> None:
    bus = EventBus()
    with _client(bus, PanelController(_agent(bus, []), bus)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_summarises_recorded_turns(tmp_path: Path) -> None:
    bus = EventBus()
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl", bus=bus, model="gemini-test")
    agent = _agent(
        bus, [LLMResponse(text="Hi.", usage=TokenUsage(input_tokens=50, output_tokens=3))]
    )
    controller = PanelController(agent, bus, metrics=recorder)
    with (
        _client(bus, controller) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        ws.receive_json()  # hello
        ws.send_json({"cmd": "text", "text": "hello"})
        metrics_event = _receive_until(ws, lambda e: e["type"] == "metrics")[-1]
        summary = client.get("/metrics").json()

    assert metrics_event["data"]["turn"]["source"] == "panel"
    assert metrics_event["data"]["session"]["input_tokens"] == 50
    assert summary["enabled"] is True
    assert summary["turns"] == 1
    assert summary["llm_requests"] == 1
    assert (summary["input_tokens"], summary["output_tokens"]) == (50, 3)
    assert "llm_total" in summary["stages_ms"]
    assert summary["session"]["turns"] == 1


def test_metrics_endpoint_without_a_recorder() -> None:
    bus = EventBus()
    with _client(bus, PanelController(_agent(bus, []), bus)) as client:
        assert client.get("/metrics").json() == {"enabled": False, "turns": 0}


def test_index_serves_the_panel_with_locked_down_headers() -> None:
    bus = EventBus()
    with _client(bus, PanelController(_agent(bus, []), bus)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<canvas id="orb"' in response.text
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "ws://127.0.0.1:8000" in policy
    assert response.headers["x-frame-options"] == "DENY"


def test_panel_page_has_no_external_dependencies() -> None:
    page = INDEX_HTML.read_text(encoding="utf-8").lower()
    for forbidden in ("http://", "https://", "<script src", "stylesheet", "@import", "cdn"):
        assert forbidden not in page, forbidden
    assert "new websocket(" in page  # connects back to its own origin only


@pytest.mark.parametrize("path", ["/", "/health"])
def test_requests_for_a_foreign_host_are_refused(path: str) -> None:
    bus = EventBus()
    with _client(
        bus, PanelController(_agent(bus, []), bus), base_url="http://evil.example:8000"
    ) as client:
        response = client.get(path)
    assert response.status_code == 403  # e.g. a DNS-rebinding attempt


# --- WebSocket -----------------------------------------------------------------


def test_websocket_says_hello_then_streams_published_events() -> None:
    bus = EventBus()
    controller = PanelController(_agent(bus, []), bus, voice_error="no mic")
    with (
        _client(bus, controller) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        hello = ws.receive_json()
        assert hello["type"] == HELLO
        assert hello["data"]["state"] == "idle"
        assert hello["data"]["voice"] is False
        assert hello["data"]["voice_error"] == "no mic"

        bus.emit(REPLY, text="from the bus")  # published from this (non-loop) thread

        event = ws.receive_json()
        assert (event["type"], event["data"]) == (REPLY, {"text": "from the bus"})
        assert isinstance(event["ts"], float)


def test_disconnect_unsubscribes() -> None:
    bus = EventBus()
    with _client(bus, PanelController(_agent(bus, []), bus)) as client:
        with client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws:
            ws.receive_json()
            assert bus.has_subscribers
        deadline = time.monotonic() + 2
        while bus.has_subscribers and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not bus.has_subscribers


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://evil.example:8000",  # DNS rebinding: right port, wrong name
        "http://127.0.0.1:3000",  # another local dev server
        "null",  # sandboxed iframe / file:// page
    ],
)
def test_websocket_from_another_origin_is_refused(origin: str) -> None:
    bus = EventBus()
    with (
        _client(bus, PanelController(_agent(bus, []), bus)) as client,
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect(WS_URL, headers={"origin": origin}),
    ):
        pass
    assert refused.value.code == 1008
    assert not bus.has_subscribers


def test_websocket_with_a_foreign_host_is_refused() -> None:
    bus = EventBus()
    with (
        _client(bus, PanelController(_agent(bus, []), bus)) as client,
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("ws://evil.example:8000/ws", headers={"origin": BASE}),
    ):
        pass


def test_text_command_runs_the_agent_and_streams_the_whole_turn() -> None:
    bus = EventBus()
    agent = _agent(
        bus,
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall("t1", "echo", {"text": "hi"})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Done."),
        ],
        RecordingTool(),
    )
    with (
        _client(bus, PanelController(agent, bus)) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        ws.receive_json()  # hello
        ws.send_json({"cmd": "text", "text": "  echo hi  "})
        events = _receive_until(ws, _is_idle)

    assert _summary(events) == [
        (TRANSCRIPT, "echo hi"),
        (STATE, "thinking"),
        (TOOL, "started"),
        (TOOL, "completed"),
        (REPLY, "Done."),
        (STATE, "idle"),
    ]
    assert events[0]["data"]["source"] == "text"
    assert events[2]["data"]["name"] == "echo"
    # Exactly what text mode sends: the stripped command as the user message.
    assert _llm(agent).requests[0][0][0].content == "echo hi"


def test_bad_messages_get_an_error_and_the_socket_survives() -> None:
    bus = EventBus()
    agent = _agent(bus, [LLMResponse(text="Still here.")])
    with (
        _client(bus, PanelController(agent, bus)) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        ws.receive_json()  # hello
        ws.send_text("not json")
        ws.send_bytes(b"\x00\x01")
        ws.send_json(["not", "an", "object"])
        ws.send_json({"cmd": "dance"})
        ws.send_json({"cmd": "text", "text": "   "})
        ws.send_json({"cmd": "text", "text": 42})
        ws.send_json({"cmd": "text", "text": "x" * (MAX_TEXT_CHARS + 1)})
        errors = [ws.receive_json() for _ in range(7)]
        assert all(e["type"] == ERROR for e in errors), errors

        ws.send_json({"cmd": "text", "text": "still there?"})
        events = _receive_until(ws, _is_reply)
    assert events[-1]["data"]["text"] == "Still here."


def test_typed_exit_never_shuts_jarvis_down() -> None:
    bus = EventBus()
    agent = _agent(bus, [])
    with (
        _client(bus, PanelController(agent, bus)) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        ws.receive_json()
        ws.send_json({"cmd": "text", "text": "Exit."})
        notice = ws.receive_json()
    assert (notice["type"], notice["data"]["message"]) == (NOTICE, EXIT_HINT)
    assert _llm(agent).requests == []


def test_talk_without_voice_is_refused_with_the_reason() -> None:
    bus = EventBus()
    controller = PanelController(_agent(bus, []), bus, voice_error="JARVIS_TTS_VOICE is not set")
    with (
        _client(bus, controller) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        ws.receive_json()
        ws.send_json({"cmd": "talk"})
        event = ws.receive_json()
    assert event["type"] == ERROR
    assert "JARVIS_TTS_VOICE is not set" in event["data"]["message"]


def test_talk_runs_one_push_to_talk_turn() -> None:
    bus = EventBus()
    agent = _agent(bus, [LLMResponse(text="It's noon.")])
    voice, mic, tts = _push_to_talk(agent, bus, transcripts=["What time is it?"])
    with (
        _client(bus, PanelController(agent, bus, voice=voice)) as client,
        client.websocket_connect(WS_URL, headers=PANEL_ORIGIN) as ws,
    ):
        hello = ws.receive_json()
        assert hello["data"]["voice"] is True and hello["data"]["hands_free"] is False
        ws.send_json({"cmd": "talk"})
        events = _receive_until(ws, _is_reply)
        events += _receive_until(ws, _is_idle)  # the loop settles...
        events += _receive_until(ws, _is_idle)  # ...then the controller, after closing the mic

    assert [s for s in _summary(events) if s[0] != STATE or s[1] != "idle"] == [
        (STATE, "listening"),
        (STATE, "thinking"),
        (TRANSCRIPT, "What time is it?"),
        (REPLY, "It's noon."),
        (STATE, "speaking"),
    ]
    assert tts.spoken == ["It's noon."]
    assert (mic.starts, mic.closes) == (1, 1)  # the mic is open only during the turn


# --- controller (no server) ----------------------------------------------------


def test_busy_panel_refuses_new_requests_until_done() -> None:
    class GatedLLM(FakeLLMClient):
        def __init__(self) -> None:
            super().__init__([])
            self.gate = asyncio.Event()

        async def complete(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None
        ) -> LLMResponse:
            await self.gate.wait()
            return LLMResponse(text="first done")

    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()
        llm = GatedLLM()
        agent = Agent(llm, ToolRegistry(), events=bus)
        voice, _mic, _tts = _push_to_talk(agent, bus, transcripts=[])
        controller = PanelController(agent, bus, voice=voice)

        assert controller.handle({"cmd": "text", "text": "first"}) is None
        await _wait_for(queue, _state("thinking"))
        for second in ({"cmd": "text", "text": "second"}, {"cmd": "talk"}):
            response = controller.handle(second)
            assert response is not None and response.data["message"] == BUSY_MESSAGE
        assert controller.snapshot()["state"] == "thinking"

        llm.gate.set()
        events = await _wait_for(queue, _state("idle"))
        assert [e.data["text"] for e in events if e.type == REPLY] == ["first done"]
        assert not controller.busy
        await controller.aclose()

    asyncio.run(scenario())


def test_stop_abandons_a_push_to_talk_recording() -> None:
    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()
        agent = _agent(bus, [])
        voice, mic, tts = _push_to_talk(agent, bus, transcripts=[], recorder=EndlessRecorder())
        controller = PanelController(agent, bus, voice=voice)

        assert controller.handle({"cmd": "talk"}) is None
        await _wait_for(queue, _state("listening"))
        stopped = controller.handle({"cmd": "stop"})
        assert stopped is not None and stopped.data["message"] == "Stopped listening."
        await _wait_for(queue, _state("idle"))

        assert _llm(agent).requests == [] and tts.spoken == []
        assert (mic.starts, mic.closes) == (1, 1)
        nothing = controller.handle({"cmd": "stop"})
        assert nothing is not None and nothing.type == NOTICE
        await controller.aclose()

    asyncio.run(scenario())


def _hands_free(
    agent: Agent, bus: EventBus, *, transcripts: list[str], recorder: Any = None
) -> tuple[HandsFreeVoice, FakeTTSEngine, SilentDetector]:
    detector, tts = SilentDetector(), FakeTTSEngine()
    trigger = ManualWakeTrigger(detector)
    loop = WakeWordLoop(
        agent,
        FakeSTTEngine(transcripts),
        tts,
        trigger,
        recorder or FakeVAD(),
        SlowSource(),
        display=lambda _line: None,
        events=bus,
    )
    return HandsFreeVoice(loop, trigger, bus), tts, detector


def test_hands_free_talk_button_fires_the_wake_loop() -> None:
    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()
        agent = _agent(bus, [LLMResponse(text="Hello!")])
        voice, tts, detector = _hands_free(agent, bus, transcripts=["hi jarvis"])
        controller = PanelController(agent, bus, voice=voice)
        await controller.start()
        await _wait_for(queue, _state("idle"))
        assert controller.snapshot()["hands_free"] is True

        assert controller.handle({"cmd": "talk"}) is None
        events = await _wait_for(queue, lambda e: e.type == REPLY)

        assert [e.data["state"] for e in events if e.type == STATE] == ["listening", "thinking"]
        await _wait_for(queue, _state("idle"))
        assert tts.spoken == ["Hello!"]
        assert detector.resets == 1  # reset after the (manual) trigger, as after a real one
        assert voice.state == "idle"  # still listening for the wake word
        await controller.aclose()
        assert voice.state is None

    asyncio.run(scenario())


def test_hands_free_spoken_exit_keeps_the_panel_running() -> None:
    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()
        agent = _agent(bus, [])
        voice, _tts, _detector = _hands_free(agent, bus, transcripts=["exit"])
        controller = PanelController(agent, bus, voice=voice)
        await controller.start()
        await _wait_for(queue, _state("idle"))

        controller.handle({"cmd": "talk"})
        await _wait_for(queue, lambda e: e.type == NOTICE and e.data["message"] == EXIT_HINT)
        await _wait_for(queue, _state("idle"))  # listening for the wake word again

        assert voice.state == "idle"
        assert _llm(agent).requests == []
        await controller.aclose()

    asyncio.run(scenario())


def test_hands_free_stop_abandons_the_command_and_keeps_listening() -> None:
    async def scenario() -> None:
        bus = EventBus()
        queue = bus.subscribe()
        agent = _agent(bus, [])
        voice, _tts, _detector = _hands_free(agent, bus, transcripts=[], recorder=EndlessRecorder())
        controller = PanelController(agent, bus, voice=voice)
        await controller.start()
        await _wait_for(queue, _state("idle"))

        controller.handle({"cmd": "talk"})
        await _wait_for(queue, _state("listening"))
        stopped = controller.handle({"cmd": "stop"})
        assert stopped is not None and stopped.data["message"] == "Stopped listening."
        await _wait_for(queue, _state("idle"))

        assert voice.state == "idle"  # restarted, not dead
        assert _llm(agent).requests == []
        await controller.aclose()

    asyncio.run(scenario())


def test_manual_trigger_fires_once_and_otherwise_defers() -> None:
    inner = SilentDetector()
    trigger = ManualWakeTrigger(inner)
    frame = FakeFrameSource().read(1280)

    assert trigger.frame_samples == 1280
    assert trigger.process(frame) == 0.0
    trigger.trigger()
    assert trigger.process(frame) == 1.0
    assert trigger.process(frame) == 0.0  # only once

    trigger.trigger()
    trigger.reset()  # a pending click is dropped on reset
    assert trigger.process(frame) == 0.0
    assert inner.resets == 1
    assert ManualWakeTrigger().process(frame) == 0.0


# --- loopback only --------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5", "example.com", "", "127.0.0.1.evil.com"]
)
def test_non_loopback_hosts_are_refused(host: str) -> None:
    with pytest.raises(NonLoopbackHostError, match="loopback"):
        check_loopback(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "127.0.0.2"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    check_loopback(host)


def test_bind_refuses_before_opening_any_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_sockets(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("a socket was opened for a non-loopback host")

    monkeypatch.setattr(app_module.socket, "socket", no_sockets)
    with pytest.raises(NonLoopbackHostError):
        bind_loopback_socket("0.0.0.0", 8000)


def test_host_and_origin_rules() -> None:
    assert (
        host_allowed("127.0.0.1:8000") and host_allowed("localhost") and host_allowed("[::1]:8000")
    )
    assert not host_allowed(None) and not host_allowed("evil.example:8000")
    assert not host_allowed("127.0.0.1:notaport")

    assert origin_allowed(None, 8000)  # not a browser
    assert origin_allowed("http://127.0.0.1:8000", 8000)
    assert origin_allowed("http://localhost:8000", 8000)
    assert origin_allowed("http://127.0.0.1", 80)
    assert not origin_allowed("http://127.0.0.1:8001", 8000)
    assert not origin_allowed("https://127.0.0.1:8000", 8000)
    assert not origin_allowed("http://evil.example:8000", 8000)


def test_panel_url() -> None:
    assert panel_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
    assert panel_url("::1", 9000) == "http://[::1]:9000"


# --- CLI -----------------------------------------------------------------------


def test_serve_combines_with_wake_only(capsys: pytest.CaptureFixture[str]) -> None:
    args = parse_args(["--serve", "--wake"])
    assert args.serve and args.wake
    assert not parse_args(["--wake"]).serve
    for other in ("--voice", "--health", "--reindex", "--list-tools", "--list-devices"):
        with pytest.raises(SystemExit) as exited:
            parse_args(["--serve", other])
        assert exited.value.code == 2
    assert "--serve can only be combined with --wake" in capsys.readouterr().err
