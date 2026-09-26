"""Command-line entrypoint.

python -m jarvis.cli                 interactive text chat
python -m jarvis.cli --voice         push-to-talk voice chat
python -m jarvis.cli --wake          hands-free: say "Hey Jarvis", then your command
python -m jarvis.cli --serve         local web panel at http://127.0.0.1:8000 (type or Talk)
python -m jarvis.cli --serve --wake  the panel plus hands-free listening
python -m jarvis.cli --reindex       build/refresh the memory index, then exit
python -m jarvis.cli --list-tools    show auto-discovered tools, then exit
python -m jarvis.cli --health        check config and vault, then exit
python -m jarvis.cli --list-devices  show audio device indices, then exit
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError

from jarvis import __version__
from jarvis.config import Settings, get_settings
from jarvis.core.agent import Agent
from jarvis.core.events import LEVEL_MIC, LEVEL_SPEAKER, EventBus, LevelMeter
from jarvis.core.interfaces import (
    AudioSamples,
    CommandRecorder,
    FrameSource,
    LLMError,
    RetrievalError,
    STTEngine,
    TTSEngine,
    VoiceError,
    WakeWordDetector,
)
from jarvis.core.voice_session import EXIT_COMMANDS, TurnOutcome, VoiceSession
from jarvis.llm.factory import create_llm_client
from jarvis.logging import configure_logging, get_logger
from jarvis.memory.auto_index import AutoIndexingAgent
from jarvis.memory.factory import build_memory
from jarvis.memory.vault import Vault, VaultError
from jarvis.tools.base import DuplicateToolError, ToolRegistry, discover_tool_classes
from jarvis.tools.context import ToolContext

if TYPE_CHECKING:
    from jarvis.server.controller import PanelController

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description="Jarvis personal assistant.")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--health",
        action="store_true",
        help="load settings, check the vault is writable, and exit",
    )
    mode.add_argument(
        "--voice",
        action="store_true",
        help="push-to-talk voice chat (Enter to start/stop recording)",
    )
    mode.add_argument(
        "--wake",
        action="store_true",
        help="hands-free voice chat: wake word, then auto-stop when you stop talking",
    )
    mode.add_argument(
        "--reindex",
        action="store_true",
        help="build/refresh the memory index over the vault's Jarvis notes, then exit",
    )
    mode.add_argument(
        "--list-tools",
        action="store_true",
        help="list auto-discovered tools (category, confirmation, enabled) and exit",
    )
    mode.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio devices (for JARVIS_INPUT_DEVICE / JARVIS_OUTPUT_DEVICE) and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="open the local web panel (http://127.0.0.1:8000, this computer only); "
        "add --wake for hands-free listening too",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; `--serve` combines with `--wake` and nothing else."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.serve and any(
        (args.health, args.voice, args.reindex, args.list_tools, args.list_devices)
    ):
        parser.error("--serve can only be combined with --wake")
    return args


def _load_settings() -> Settings | None:
    """Load settings and configure logging; print a friendly error on failure."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"FAIL  config invalid; check your .env (see .env.example):\n{exc}", file=sys.stderr)
        return None
    configure_logging(settings.log_level)
    return settings


def build_agent(settings: Settings, *, events: EventBus | None = None) -> Agent:
    """Wire everything together: vault -> memory -> tools -> LLM client -> agent.

    Memory backends are created but not loaded; the embedding model and index
    open on first use, so startup stays fast. `events` (the panel's bus) makes
    the agent publish its tool calls.

    Raises:
        VaultError: If the vault path is invalid.
        RetrievalError: If a memory backend is unknown.
        LLMError: If the LLM provider is unknown or misconfigured.
    """
    vault = Vault(settings.vault_path)
    memory = build_memory(settings, vault.jarvis_root)
    registry = build_tool_registry(settings, ToolContext(settings, vault, memory))
    llm = create_llm_client(settings)
    if not settings.confirm_side_effects:
        log.warning("agent.confirmation_disabled", setting="JARVIS_CONFIRM_SIDE_EFFECTS=false")
        print(
            "WARNING: JARVIS_CONFIRM_SIDE_EFFECTS is off: tools with side effects "
            "will run WITHOUT asking you first.",
            file=sys.stderr,
        )
    if settings.auto_index:
        return AutoIndexingAgent(
            llm,
            registry,
            indexer=memory.indexer,
            max_iterations=settings.agent_max_iterations,
            confirm_side_effects=settings.confirm_side_effects,
            events=events,
        )
    return Agent(
        llm,
        registry,
        max_iterations=settings.agent_max_iterations,
        confirm_side_effects=settings.confirm_side_effects,
        events=events,
    )


def build_tool_registry(settings: Settings, context: ToolContext) -> ToolRegistry:
    """Discover every tool in `jarvis.tools`, build each from `context`, apply `enabled_tools`.

    Dependency injection: tools pull what they need from the `ToolContext`
    (vault, memory, settings, clock) in their own `from_context`, so a new
    tool file needs no change here.

    Raises:
        DuplicateToolError: If two tools share a name.
    """
    registry = ToolRegistry()
    registry.discover(context=context)
    for problem in registry.discovery_problems:
        log.warning("tools.skipped", where=problem.where, error=problem.error)
    if settings.enabled_tools is not None:
        unknown = registry.restrict(settings.enabled_tools)
        if unknown:
            log.warning("tools.unknown_in_enabled_tools", names=unknown)
    return registry


def run_health() -> int:
    """Check config and vault; print a status line. Returns a process exit code."""
    settings = _load_settings()
    if settings is None:
        return 1
    try:
        vault = Vault(settings.vault_path)
        jarvis_root = vault.check_writable()
    except (VaultError, OSError) as exc:
        log.error("health.failed", error=str(exc))
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    log.info("health.ok", vault=str(vault.vault_root), llm_provider=settings.llm_provider)
    key_status = "set" if settings.llm_api_key else "MISSING (set LLM_API_KEY)"
    tts = settings.tts_provider
    if tts == "piper":
        voice = settings.tts_voice
        tts += (
            f" ({voice})" if voice and voice.is_file() else " (voice MISSING: set JARVIS_TTS_VOICE)"
        )
    print(f"OK  jarvis {__version__}")
    print(f"    vault:   {vault.vault_root}")
    print(f"    jarvis:  {jarvis_root} (writable)")
    print(f"    llm:     {settings.llm_provider} / {settings.llm_model}, api key {key_status}")
    print(
        f"    stt:     whisper {settings.stt_model} ({settings.stt_device}, {settings.stt_compute_type})"
    )
    print(f"    tts:     {tts}")
    print(
        f"    wake:    '{settings.wake_word_model}' (threshold {settings.wake_word_threshold}), "
        f"vad {settings.vad_aggressiveness} / {settings.vad_silence_ms} ms silence / "
        f"{settings.command_max_seconds:g} s max"
    )
    try:
        indexed = build_memory(settings, vault.jarvis_root).indexer.indexed_notes()
        index_status = f"{indexed} notes indexed" if indexed else "not indexed yet (run --reindex)"
    except RetrievalError as exc:
        index_status = f"MISCONFIGURED: {exc}"
    auto = "on" if settings.auto_index else "off"
    print(
        f"    memory:  {settings.embedder_provider} {settings.embed_model} -> "
        f"{settings.vector_store} at {settings.chroma_path}, {index_status}, auto-index {auto}"
    )
    return 0


def run_reindex() -> int:
    """Build or refresh the memory index over the whole vault. Returns a process exit code."""
    settings = _load_settings()
    if settings is None:
        return 1
    try:
        vault = Vault(settings.vault_path)
        memory = build_memory(settings, vault.jarvis_root)
        print(
            f"Indexing {vault.jarvis_root} with {settings.embed_model} "
            "(the first run downloads the model, about 70 MB)..."
        )
        started = time.perf_counter()
        stats = memory.indexer.reindex_all()
        total_chunks = memory.store.count()
    except (VaultError, RetrievalError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    print(
        f"OK  indexed in {time.perf_counter() - started:.1f}s: {stats.added} added, "
        f"{stats.updated} updated, {stats.skipped} unchanged, {stats.removed} removed "
        f"({stats.chunks} chunks embedded)"
    )
    print(
        f"    index: {memory.indexer.indexed_notes()} notes, {total_chunks} chunks "
        f"at {settings.chroma_path}"
    )
    return 0


def run_list_tools() -> int:
    """Print every discovered tool and whether it's enabled. Returns a process exit code.

    Lists tool classes without building them, so it works even before the
    vault or API key is configured.
    """
    try:
        classes, problems = discover_tool_classes()
    except DuplicateToolError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    try:
        enabled = get_settings().enabled_tools
    except ValidationError:
        enabled = None  # config incomplete: list everything as enabled

    rows = [("NAME", "CATEGORY", "CONFIRM", "ENABLED", "MODULE")]
    for cls in classes:
        rows.append(
            (
                cls.name,
                cls.category,
                "yes (y/N)" if cls.requires_confirmation else "no",
                "yes" if enabled is None or cls.name in enabled else "no",
                cls.__module__.removeprefix("jarvis.tools."),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    print(f"Discovered {len(classes)} tools in jarvis.tools:")
    for row in rows:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    if enabled is not None:
        unknown = sorted(set(enabled) - {cls.name for cls in classes})
        print(
            f"\nJARVIS_ENABLED_TOOLS whitelist is active{f'; unknown names: {unknown}' if unknown else ''}."
        )
    for problem in problems:
        print(f"SKIPPED  {problem.where}: {problem.error}")
    return 0


def run_list_devices() -> int:
    """Print audio devices. Returns a process exit code."""
    from jarvis.audio.io import list_devices

    print(list_devices())
    print("\n> = default input, < = default output. Use the leading number as the device index.")
    return 0


def run_repl() -> int:
    """Interactive text chat. Returns a process exit code."""
    settings = _load_settings()
    if settings is None:
        return 1
    try:
        agent = build_agent(settings)
    except (VaultError, LLMError, RetrievalError, DuplicateToolError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    print(f"Jarvis ready ({settings.llm_provider} / {settings.llm_model}). Type 'exit' to quit.")
    # One event loop for the whole session, so the LLM client's HTTP session is reused.
    with asyncio.Runner() as runner:
        try:
            _chat(agent, runner)
        finally:
            runner.run(agent.llm.aclose())
    print("Bye.")
    return 0


def run_voice() -> int:
    """Push-to-talk voice chat. Returns a process exit code."""
    settings = _load_settings()
    if settings is None:
        return 1

    # Voice deps are imported only in voice mode, keeping text mode light.
    from jarvis.audio.io import record_until_enter
    from jarvis.stt.whisper_stt import WhisperSTT
    from jarvis.tts.factory import create_tts_engine

    try:
        agent = build_agent(settings)
        stt = WhisperSTT.from_settings(settings)
        tts = create_tts_engine(settings)
    except (VaultError, LLMError, RetrievalError, VoiceError, DuplicateToolError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    session = VoiceSession(agent, stt, tts)

    def record() -> AudioSamples:
        return record_until_enter(settings.sample_rate, settings.input_device)

    with asyncio.Runner() as runner:
        try:
            print(f"Loading speech model '{settings.stt_model}' (the first run downloads it)...")
            try:
                # Load up front so the first utterance isn't stuck behind a download.
                runner.run(asyncio.to_thread(stt.load))
            except VoiceError as exc:
                print(f"FAIL  {exc}", file=sys.stderr)
                return 1
            print(
                f"Jarvis voice ready ({settings.llm_provider} / {settings.llm_model}, "
                f"tts: {settings.tts_provider})."
            )
            _voice_chat(session, runner, record)
        finally:
            runner.run(agent.llm.aclose())
    print("Bye.")
    return 0


def run_wake() -> int:
    """Hands-free wake-word voice loop. Returns a process exit code."""
    settings = _load_settings()
    if settings is None:
        return 1

    # Voice deps are imported only in voice modes, keeping text mode light.
    from jarvis.audio.io import STREAM_SAMPLE_RATE, MicStream, play_chime
    from jarvis.audio.vad import VoiceActivityDetector
    from jarvis.core.voice_loop import WakeWordLoop
    from jarvis.stt.whisper_stt import WhisperSTT
    from jarvis.tts.factory import create_tts_engine
    from jarvis.wakeword.openwakeword_detector import OpenWakeWordDetector

    try:
        agent = build_agent(settings)
        # The shared mic stream is always 16 kHz, whatever JARVIS_SAMPLE_RATE says for --voice.
        stt = WhisperSTT.from_settings(
            settings.model_copy(update={"sample_rate": STREAM_SAMPLE_RATE})
        )
        tts = create_tts_engine(settings)
        detector = OpenWakeWordDetector(settings.wake_word_model)
        recorder = VoiceActivityDetector(
            aggressiveness=settings.vad_aggressiveness,
            frame_ms=settings.vad_frame_ms,
            silence_ms=settings.vad_silence_ms,
            max_seconds=settings.command_max_seconds,
            sample_rate=STREAM_SAMPLE_RATE,
        )
    except (VaultError, LLMError, RetrievalError, VoiceError, ValueError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    def chime() -> None:
        play_chime(settings.output_device)

    wake_phrase = _wake_phrase(settings.wake_word_model)
    with asyncio.Runner() as runner:
        try:
            print(f"Loading models (Whisper '{settings.stt_model}' downloads on first run)...")
            runner.run(asyncio.to_thread(stt.load))
            runner.run(asyncio.to_thread(detector.load))
            with MicStream(STREAM_SAMPLE_RATE, settings.input_device) as mic:
                loop = WakeWordLoop(
                    agent,
                    stt,
                    tts,
                    detector,
                    recorder,
                    mic,
                    threshold=settings.wake_word_threshold,
                    chime=chime if settings.wake_chime else None,
                    wake_phrase=wake_phrase,
                    display=_safe_print,
                )
                print(
                    f"Jarvis is listening. Say '{wake_phrase}', then your command. Ctrl-C to quit."
                )
                runner.run(loop.run())
        except KeyboardInterrupt:
            print()
        except VoiceError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 1
        finally:
            runner.run(agent.llm.aclose())
    print("Bye.")
    return 0


def run_serve(*, wake: bool = False) -> int:
    """The local web panel, optionally with hands-free listening. Returns a process exit code.

    The panel, the agent and (with `wake`) the wake-word loop share one event
    loop. The agent is the same one text mode uses, now publishing events.
    """
    settings = _load_settings()
    if settings is None:
        return 1
    bus = EventBus()
    try:
        agent = build_agent(settings, events=bus)
    except (VaultError, LLMError, RetrievalError, DuplicateToolError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    with asyncio.Runner() as runner:
        try:
            if wake:
                code = _serve_hands_free(settings, agent, bus, runner)
            else:
                code = _serve_push_to_talk(settings, agent, bus, runner)
        except KeyboardInterrupt:
            print()
            code = 0
        finally:
            runner.run(agent.llm.aclose())
    if code == 0:
        print("Bye.")
    return code


def _serve_push_to_talk(
    settings: Settings, agent: Agent, bus: EventBus, runner: asyncio.Runner
) -> int:
    """`--serve`: typing works always; Talk opens the mic for one command at a time."""
    from jarvis.server.controller import PanelController, PushToTalkVoice

    voice: PushToTalkVoice | None = None
    voice_error: str | None = None
    try:
        # Voice deps are imported only when voice is used, keeping text mode light.
        from jarvis.audio.io import STREAM_SAMPLE_RATE, MicStream
        from jarvis.core.voice_loop import WakeWordLoop

        stt, tts, recorder, chime = _build_panel_voice_io(settings, bus)
        mic = MicStream(
            STREAM_SAMPLE_RATE, settings.input_device, audio_listener=LevelMeter(bus, LEVEL_MIC)
        )

        def make_loop(detector: WakeWordDetector, source: FrameSource) -> WakeWordLoop:
            # No wake word in this mode, so the loop's "listening for ..." lines stay quiet.
            return WakeWordLoop(
                agent, stt, tts, detector, recorder, source, chime=chime, display=_quiet, events=bus
            )

        voice = PushToTalkVoice(make_loop, mic, bus, warm_up=stt.load)
    except (VoiceError, ValueError, OSError, ImportError) as exc:
        voice_error = str(exc)
        print(f"NOTE  Talk is disabled (typing still works): {exc}", file=sys.stderr)
    controller = PanelController(agent, bus, voice=voice, voice_error=voice_error)
    hint = "Type a command or press Talk." if voice else "Type a command."
    return _run_panel(settings, bus, controller, runner, hint)


def _serve_hands_free(
    settings: Settings, agent: Agent, bus: EventBus, runner: asyncio.Runner
) -> int:
    """`--serve --wake`: the `--wake` loop, always listening, with the panel on top."""
    from jarvis.audio.io import STREAM_SAMPLE_RATE, MicStream
    from jarvis.core.voice_loop import WakeWordLoop
    from jarvis.server.controller import HandsFreeVoice, ManualWakeTrigger, PanelController
    from jarvis.wakeword.openwakeword_detector import OpenWakeWordDetector

    try:
        stt, tts, recorder, chime = _build_panel_voice_io(settings, bus)
        detector = OpenWakeWordDetector(settings.wake_word_model)
        print(f"Loading models (Whisper '{settings.stt_model}' downloads on first run)...")
        runner.run(asyncio.to_thread(stt.load))
        runner.run(asyncio.to_thread(detector.load))
    except (VoiceError, ValueError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    wake_phrase = _wake_phrase(settings.wake_word_model)
    trigger = ManualWakeTrigger(detector)  # the panel's Talk button fires it too
    with MicStream(
        STREAM_SAMPLE_RATE, settings.input_device, audio_listener=LevelMeter(bus, LEVEL_MIC)
    ) as mic:
        loop = WakeWordLoop(
            agent,
            stt,
            tts,
            trigger,
            recorder,
            mic,
            threshold=settings.wake_word_threshold,
            chime=chime,
            wake_phrase=wake_phrase,
            display=_safe_print,
            events=bus,
        )
        voice = HandsFreeVoice(loop, trigger, bus, wake_phrase=wake_phrase)
        controller = PanelController(agent, bus, voice=voice)
        return _run_panel(
            settings, bus, controller, runner, f"Say '{wake_phrase}', type, or press Talk."
        )


def _build_panel_voice_io(
    settings: Settings, bus: EventBus
) -> tuple[STTEngine, TTSEngine, CommandRecorder, Callable[[], None] | None]:
    """STT, TTS (metered for the orb), the end-of-speech recorder and the wake chime.

    Raises:
        VoiceError / ValueError: If a voice setting is invalid.
    """
    from jarvis.audio.io import STREAM_SAMPLE_RATE, play_chime
    from jarvis.audio.vad import VoiceActivityDetector
    from jarvis.stt.whisper_stt import WhisperSTT
    from jarvis.tts.factory import create_tts_engine

    # The shared mic stream is always 16 kHz, whatever JARVIS_SAMPLE_RATE says for --voice.
    stt = WhisperSTT.from_settings(settings.model_copy(update={"sample_rate": STREAM_SAMPLE_RATE}))
    tts = create_tts_engine(settings, level_listener=LevelMeter(bus, LEVEL_SPEAKER))
    recorder = VoiceActivityDetector(
        aggressiveness=settings.vad_aggressiveness,
        frame_ms=settings.vad_frame_ms,
        silence_ms=settings.vad_silence_ms,
        max_seconds=settings.command_max_seconds,
        sample_rate=STREAM_SAMPLE_RATE,
    )

    def chime() -> None:
        play_chime(settings.output_device)

    return stt, tts, recorder, chime if settings.wake_chime else None


def _run_panel(
    settings: Settings,
    bus: EventBus,
    controller: "PanelController",
    runner: asyncio.Runner,
    hint: str,
) -> int:
    """Bind the loopback port, print the URL and serve until Ctrl-C."""
    from jarvis.server.app import NonLoopbackHostError, bind_loopback_socket, create_app, panel_url
    from jarvis.server.app import serve as serve_panel

    try:
        sock = bind_loopback_socket(settings.ui_host, settings.ui_port)
    except NonLoopbackHostError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"FAIL  can't listen on {settings.ui_host}:{settings.ui_port} ({exc}). Is Jarvis "
            "already running? Set JARVIS_UI_PORT to use another port.",
            file=sys.stderr,
        )
        return 1
    app = create_app(bus, controller, port=settings.ui_port)
    print(f"Jarvis panel: {panel_url(settings.ui_host, settings.ui_port)}  (this computer only)")
    print(f"{hint} Side-effect actions ask y/N here in this terminal. Ctrl-C to quit.")
    runner.run(serve_panel(app, sock, log_level=settings.log_level))
    return 0


def _quiet(_line: str) -> None:
    """A `display` that shows nothing (the panel is the display)."""


def _wake_phrase(model: str) -> str:
    """'hey_jarvis' -> 'Hey Jarvis'; a model path -> its file stem, prettified."""
    stem = model.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".onnx")
    return stem.replace("_", " ").title()


def _safe_print(text: str) -> None:
    """Print, falling back to ASCII if the console can't show emoji."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def _chat(agent: Agent, runner: asyncio.Runner) -> None:
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in EXIT_COMMANDS:
            return
        try:
            reply = runner.run(agent.run(line))
        except KeyboardInterrupt:
            print("\n(interrupted)")
            return
        except LLMError as exc:
            print(f"jarvis> [LLM error] {exc}")
            continue
        except Exception as exc:
            log.exception("repl.turn_failed")
            print(f"jarvis> [error] {type(exc).__name__}: {exc}")
            continue
        print(f"jarvis> {reply}")


def _voice_chat(
    session: VoiceSession, runner: asyncio.Runner, record: Callable[[], AudioSamples]
) -> None:
    # Push-to-talk stays manual on purpose; the hands-free loop is `run_wake`.
    while True:
        try:
            typed = input("\nPress Enter to talk (or type a message, or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if typed:
                outcome = runner.run(session.respond(typed))
            else:
                outcome = runner.run(session.run_turn(record()))
        except KeyboardInterrupt:
            print("\n(interrupted)")
            return
        except LLMError as exc:
            print(f"jarvis> [LLM error] {exc}")
            continue
        except Exception as exc:
            log.exception("voice.turn_failed")
            print(f"jarvis> [error] {type(exc).__name__}: {exc}")
            continue
        if outcome is TurnOutcome.EXIT:
            return


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    args = parse_args(argv)
    if args.serve:
        return run_serve(wake=args.wake)
    if args.health:
        return run_health()
    if args.list_devices:
        return run_list_devices()
    if args.list_tools:
        return run_list_tools()
    if args.reindex:
        return run_reindex()
    if args.voice:
        return run_voice()
    if args.wake:
        return run_wake()
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
