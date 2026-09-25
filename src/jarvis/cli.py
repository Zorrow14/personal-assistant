"""Command-line entrypoint.

    python -m jarvis.cli            interactive text chat
    python -m jarvis.cli --health   check config and vault, then exit
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from jarvis import __version__
from jarvis.config import Settings, get_settings
from jarvis.core.agent import Agent
from jarvis.core.interfaces import LLMError
from jarvis.llm.factory import create_llm_client
from jarvis.logging import configure_logging, get_logger
from jarvis.memory.vault import Vault, VaultError
from jarvis.tools.base import ToolRegistry
from jarvis.tools.vault_tools import WriteTaskNoteTool

EXIT_COMMANDS = frozenset({"exit", "quit"})

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description="Jarvis personal assistant.")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    parser.add_argument(
        "--health",
        action="store_true",
        help="load settings, check the vault is writable, and exit",
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
    print(f"OK  jarvis {__version__}")
    print(f"    vault:   {vault.vault_root}")
    print(f"    jarvis:  {jarvis_root} (writable)")
    print(f"    llm:     {settings.llm_provider} / {settings.llm_model}, api key {key_status}")
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


def _chat(agent: Agent, runner: asyncio.Runner) -> None:
    # TODO(phase-2): voice loop (wake word -> STT -> Agent -> TTS) alongside this text loop.
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.health:
        return run_health()
    return run_repl()


if __name__ == "__main__":
    raise SystemExit(main())
