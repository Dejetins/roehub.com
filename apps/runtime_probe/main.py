from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roehub-runtime-probe")
    parser.add_argument("--role", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--entrypoint", action="append", default=[])
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--once", action="store_true")
    return parser


def _module_name(path: str) -> str:
    normalized = path.removesuffix(".py").replace("/", ".")
    if normalized.endswith(".__init__"):
        normalized = normalized.removesuffix(".__init__")
    return normalized


def _import_entrypoints(paths: list[str]) -> list[str]:
    modules: list[str] = []
    for path in paths:
        module = _module_name(path)
        importlib.import_module(module)
        modules.append(module)
    return modules


def _dependency_status(dependencies: list[str]) -> tuple[bool, dict[str, str]]:
    status: dict[str, str] = {}
    ready = True
    for dependency in dependencies:
        host, raw_port = dependency.rsplit(":", 1)
        try:
            addresses = socket.getaddrinfo(host, int(raw_port), type=socket.SOCK_STREAM)
            with socket.create_connection((host, int(raw_port)), timeout=2.0):
                pass
            status[dependency] = f"ready:{addresses[0][4][0]}"
        except (OSError, ValueError) as error:
            ready = False
            status[dependency] = f"unavailable:{type(error).__name__}"
    return ready, status


def _record_boot(*, path: Path, role: str, modules: list[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}
    payload = {
        "schema": "io.roehub.runtime-probe-state/v1alpha1",
        "role": role,
        "boots": int(previous.get("boots", 0)) + 1,
        "entrypoint_modules": modules,
        "uid": os.getuid(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


class _Server(ThreadingHTTPServer):
    role: str
    dependencies: list[str]
    state: dict[str, Any]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        assert isinstance(server, _Server)
        if self.path == "/health/live":
            self._json(HTTPStatus.OK, {"live": True, "role": server.role})
            return
        if self.path == "/health/ready":
            ready, dependencies = _dependency_status(server.dependencies)
            self._json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {"ready": ready, "role": server.role, "dependencies": dependencies},
            )
            return
        if self.path == "/state":
            self._json(HTTPStatus.OK, server.state)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.entrypoint:
        raise ValueError("at least one --entrypoint is required")
    modules = _import_entrypoints(list(args.entrypoint))
    ready, dependencies = _dependency_status(list(args.dependency))
    state = _record_boot(path=Path(args.state_path), role=args.role, modules=modules)
    if args.once:
        print(
            json.dumps(
                {"ready": ready, "dependencies": dependencies, "state": state},
                sort_keys=True,
            )
        )
        return 0 if ready else 1
    if args.port is None or args.port <= 0:
        raise ValueError("--port must be positive for service mode")
    if not ready:
        raise RuntimeError(f"dependencies are unavailable: {dependencies}")
    server = _Server(("0.0.0.0", args.port), _Handler)
    server.role = args.role
    server.dependencies = list(args.dependency)
    server.state = state
    stop = threading.Event()

    def _shutdown(*_args: object) -> None:
        if stop.is_set():
            return
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
