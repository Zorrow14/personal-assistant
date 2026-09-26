"""Best-effort desktop notifications through plyer (an optional dependency).

The only module that knows about plyer. Everything degrades quietly: if plyer
is missing or the OS refuses, the reminder is still spoken and shown in the panel.
"""

from jarvis.logging import get_logger

MAX_MESSAGE_CHARS = 250
"""Windows notification text is capped at about 256 characters."""

log = get_logger(__name__)


def desktop_notify(title: str, message: str) -> bool:
    """Show a desktop notification. Blocking; never raises.

    Returns:
        Whether the notification was handed to the OS.
    """
    try:
        from plyer import notification  # optional; imported only when a toast is wanted
    except ImportError:
        log.info("notify.unavailable", hint="install plyer (uv add plyer) for desktop toasts")
        return False
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[: MAX_MESSAGE_CHARS - 1] + "…"
    try:
        notification.notify(title=title, message=message, app_name="Jarvis", timeout=10)
    except Exception as exc:
        log.warning("notify.failed", error=f"{type(exc).__name__}: {exc}")
        return False
    return True
