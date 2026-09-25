"""Command-line entrypoint.

    python -m jarvis.cli                 interactive text chat
    python -m jarvis.cli --voice         push-to-talk voice chat
    python -m jarvis.cli --health        check config and vault, then exit
    python -m jarvis.cli --list-devices  show audio device indices, then exit
"""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from jarvis import __version__
from jarvis.config import Settings, get_settings
from jarvis.core.agent import Agent
from jarvis.core.interfaces import AudioSamples, LLMError, VoiceError
from jarvis.core.voice_session import EXIT_COMMANDS, TurnOutcome, VoiceSession
from jarvis.llm.factory import create_llm_client
from jarvis.logging import configure_logging, get_logger
from jarvis.memory.vault import Vault, VaultError
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool

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
        "--list-devices",
        action="store_true",
        help="list audio devices (for JARVIS_INPUT_DEVICE / JARVIS_OUTPUT_DEVICE) and exit",
    )
    return parser


def _load_settings() -> Settings | None:
    """Load settings and configure logging; print a friendly error on failure."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"FAIL  config invalid; check your .env (see .env.example):\n{exc}", file=sys.stderr)
        return None
    configure_logging(settings.log_level)
    return settings


def build_agent(settings: Settings) -> Agent:
    """Wire everything together: vault -> tools -> LLM client -> agent.

    Raises:
        VaultError: If the vault path is invalid.
        LLMError: If the LLM provider is unknown or misconfigured.
    """
    vault = Vault(settings.vault_path)
    registry = ToolRegistry()
    registry.register(WriteTaskNoteTool(vault))
    llm = create_llm_client(settings)
    return Agent(llm, registry, max_iterations=settings.agent_max_iterations)


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
        tts += f" ({voice})" if voice and voice.is_file() else " (voice MISSING: set JARVIS_TTS_VOICE)"
    print(f"OK  jarvis {__version__}")
    print(f"    vault:   {vault.vault_root}")
    print(f"    jarvis:  {jarvis_root} (writable)")
    print(f"    llm:     {settings.llm_provider} / {settings.llm_model}, api key {key_status}")
    print(f"    stt:     whisper {settings.stt_model} ({settings.stt_device}, {settings.stt_compute_type})")
    print(f"    tts:     {tts}")
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
    except (VaultError, LLMError) as exc:
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
    except (VaultError, LLMError, VoiceError) as exc:
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
    # TODO(phase-3): wake word + always-listening loop, and VAD auto-stop on silence.
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
    args = _build_parser().parse_args(argv)
    if args.health:
        return run_health()
    if args.list_devices:
        return run_list_devices()
    if args.voice:
        return run_voice()
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
