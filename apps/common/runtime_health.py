"""Small HTTP liveness/readiness boundary for standalone runtime roles."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeHealthState:
    service: str
    ready: bool
    mode: str
    reason: str

    def payload(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "ready": self.ready,
            "mode": self.mode,
            "reason": self.reason,
        }


class RuntimeHealthServer:
    def __init__(self, *, host: str, port: int, state: RuntimeHealthState) -> None:
        if not 0 < port < 65536:
            raise ValueError("runtime health port is invalid")
        self._state = state
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._thread: threading.Thread | None = None

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        state = self._state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health/live":
                    self._write(
                        HTTPStatus.OK,
                        {"service": state.service, "status": "alive"},
                    )
                    return
                if self.path == "/health/ready":
                    status = HTTPStatus.OK if state.ready else HTTPStatus.SERVICE_UNAVAILABLE
                    self._write(status, state.payload())
                    return
                self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def _write(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                body = (json.dumps(payload, sort_keys=True) + "\n").encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args
                return

        return Handler

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("runtime health server already started")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


__all__ = ["RuntimeHealthServer", "RuntimeHealthState"]
