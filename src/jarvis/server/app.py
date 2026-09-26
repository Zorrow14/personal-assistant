"""The local web panel: FastAPI on a loopback address, streaming events over a WebSocket.

Jarvis runs tools and can hear the microphone, so the panel is local-only by
design. There is no login, and the following checks are why that's acceptable:

- `bind_loopback_socket` refuses any non-loopback address before opening a socket.
- Every request must carry a loopback Host header, which defeats DNS rebinding.
- A browser WebSocket must come from the panel's own origin. Browsers don't
  apply CORS to WebSockets, so this check is what stops another website open in
  the same browser from connecting and driving Jarvis.

The assistant and the server share one event loop: the voice loop and agent
publish to the `EventBus`, and each open panel gets its own subscriber queue.
"""

import asyncio
import json
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from jarvis.config import is_loopback_host
from jarvis.core.events import ERROR, HELLO, Event, EventBus, offer
from jarvis.logging import get_logger
from jarvis.server.controller import PanelController

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
MAX_MESSAGE_BYTES = 64 * 1024
"""Largest inbound WebSocket message; typed commands are far smaller."""

log = get_logger(__name__)


class NonLoopbackHostError(ValueError):
    """The panel was asked to listen somewhere other machines could reach."""


def check_loopback(host: str) -> None:
    """Refuse anything but a loopback address.

    Raises:
        NonLoopbackHostError: For 0.0.0.0, LAN addresses, public names, etc.
    """
    if not is_loopback_host(host):
        raise NonLoopbackHostError(
            f"refusing to serve the panel on {host!r}: only loopback addresses "
            "(127.0.0.1, ::1, localhost) are allowed, because Jarvis runs tools and "
            "hears the mic. Set JARVIS_UI_HOST=127.0.0.1."
        )


def panel_url(host: str, port: int) -> str:
    """The URL to open in a browser, e.g. http://127.0.0.1:8000."""
    host = host.strip()
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def bind_loopback_socket(host: str, port: int) -> socket.socket:
    """Open the listening socket on a loopback address, checked before anything is bound.

    Raises:
        NonLoopbackHostError: If `host` isn't loopback (nothing is opened).
        OSError: If the port is taken or can't be bound.
    """
    check_loopback(host)
    address = "127.0.0.1" if host.strip().lower() == "localhost" else host.strip()
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            # Stop any other program binding the same port and intercepting the panel.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
    except OSError:
        sock.close()
        raise
    return sock


async def serve(app: FastAPI, sock: socket.socket, *, log_level: str = "info") -> None:
    """Run uvicorn on an already-bound loopback socket until Ctrl-C."""
    import uvicorn  # server-only dependency; imported when actually serving

    config = uvicorn.Config(
        app,
        log_level=log_level.lower(),
        access_log=False,
        lifespan="on",
        ws_max_size=MAX_MESSAGE_BYTES,
        proxy_headers=False,  # nothing sits in front of a loopback-only server
        server_header=False,
    )
    try:
        await uvicorn.Server(config).serve(sockets=[sock])
    finally:
        sock.close()


def host_allowed(host_header: str | None) -> bool:
    """Whether a Host header names this machine's loopback interface."""
    if not host_header:
        return False
    try:
        parts = urlsplit(f"//{host_header}")
        _ = parts.port  # rejects malformed ports
    except ValueError:
        return False
    return parts.hostname is not None and is_loopback_host(parts.hostname)


def origin_allowed(origin: str | None, port: int) -> bool:
    """Whether a WebSocket handshake's Origin is the panel itself.

    No Origin means a non-browser client (a script on this machine), which could
    reach the loopback port anyway. Browsers always send one.
    """
    if origin is None:
        return True
    try:
        parts = urlsplit(origin)
        origin_port = parts.port or (80 if parts.scheme == "http" else None)
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname is not None
        and is_loopback_host(parts.hostname)
        and origin_port == port
    )


class LocalOnlyMiddleware:
    """Refuses requests with a non-loopback Host, and WebSockets from other origins."""

    def __init__(self, app: ASGIApp, *, port: int) -> None:
        self.app = app
        self._port = port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = Headers(scope=scope)
            allowed = host_allowed(headers.get("host")) and (
                scope["type"] == "http" or origin_allowed(headers.get("origin"), self._port)
            )
            if not allowed:
                log.warning(
                    "server.request_refused",
                    kind=scope["type"],
                    path=scope.get("path"),
                    host=headers.get("host"),
                    origin=headers.get("origin"),
                )
                if scope["type"] == "http":
                    response = PlainTextResponse(
                        "Forbidden: this panel only answers local requests.", 403
                    )
                    await response(scope, receive, send)
                else:
                    await send({"type": "websocket.close", "code": 1008})  # refuse the handshake
                return
        await self.app(scope, receive, send)


def create_app(bus: EventBus, controller: PanelController, *, port: int) -> FastAPI:
    """Build the panel app.

    Args:
        bus: Events to stream to every connected panel.
        controller: Handles the panel's commands.
        port: The port being served; WebSocket origins must match it.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        bus.attach()  # events from audio/worker threads are handed to this loop
        await controller.start()
        try:
            yield
        finally:
            await controller.aclose()

    app = FastAPI(
        title="Jarvis panel",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(LocalOnlyMiddleware, port=port)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(
            INDEX_HTML, media_type="text/html; charset=utf-8", headers=_page_headers(port)
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_summary() -> dict[str, Any]:
        """Latency percentiles per stage, LLM requests and tokens (from the local JSONL)."""
        recorder = controller.metrics
        if recorder is None:
            return {"enabled": False, "turns": 0}
        summary = await asyncio.to_thread(recorder.summary)
        summary["enabled"] = recorder.enabled
        return summary

    @app.websocket("/ws")
    async def events_socket(websocket: WebSocket) -> None:
        await _serve_socket(websocket, bus, controller)

    # TODO(phase-6+): a Tauri shell or Next.js front end can wrap this same page; its
    # origin (e.g. tauri://localhost) would then need adding to `origin_allowed`.
    return app


async def _serve_socket(websocket: WebSocket, bus: EventBus, controller: PanelController) -> None:
    """Forward every bus event to this panel, and act on the commands it sends."""
    await websocket.accept()
    queue = bus.subscribe()
    offer(queue, Event(HELLO, controller.snapshot()))
    sender = asyncio.create_task(_forward(websocket, queue), name="jarvis-panel-sender")
    log.info("server.panel_connected")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            response = _handle_message(message, controller)
            if response is not None:
                offer(queue, response)  # to this panel only
    finally:
        bus.unsubscribe(queue)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        log.info("server.panel_disconnected")


async def _forward(websocket: WebSocket, queue: asyncio.Queue[Event]) -> None:
    while True:
        event = await queue.get()
        await websocket.send_text(event.to_json())


def _handle_message(message: Message, controller: PanelController) -> Event | None:
    text = message.get("text")
    if text is None:
        return Event(ERROR, {"message": "Send commands as JSON text messages."})
    try:
        command: Any = json.loads(text)
    except json.JSONDecodeError:
        return Event(ERROR, {"message": "That message wasn't valid JSON."})
    try:
        return controller.handle(command)
    except Exception as exc:  # a bad message must never take the socket down
        log.exception("server.command_failed")
        return Event(ERROR, {"message": f"{type(exc).__name__}: {exc}"})


def _page_headers(port: int) -> dict[str, str]:
    sockets = " ".join(f"ws://{host}:{port}" for host in ("127.0.0.1", "localhost", "[::1]"))
    policy = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        f"img-src data:; connect-src 'self' {sockets}; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )
    return {
        "Content-Security-Policy": policy,
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
    }
