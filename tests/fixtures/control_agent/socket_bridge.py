from __future__ import annotations

import select
import socket
import socketserver
from pathlib import Path
from threading import Thread
from typing import Any


def _relay(left: socket.socket, right: socket.socket) -> None:
    peers = (left, right)
    while True:
        readable, _, _ = select.select(peers, (), (), 30)
        if not readable:
            continue
        for source in readable:
            data = source.recv(64 * 1024)
            if not data:
                return
            destination = right if source is left else left
            destination.sendall(data)


def start_unix_to_tcp_bridge(
    unix_path: Path,
) -> tuple[socketserver.ThreadingTCPServer, Thread, int]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                upstream.connect(str(unix_path))
                _relay(self.request, upstream)

    server = socketserver.ThreadingTCPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


def start_tcp_to_unix_bridge(
    *,
    host: str,
    port: int,
    unix_path: Path,
) -> tuple[socketserver.ThreadingUnixStreamServer, Thread]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            with socket.create_connection((host, port), timeout=10) as upstream:
                _relay(self.request, upstream)

    unix_path.unlink(missing_ok=True)
    server = socketserver.ThreadingUnixStreamServer(str(unix_path), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_bridge(server: Any, thread: Thread, *, unix_path: Path | None = None) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    if unix_path is not None:
        unix_path.unlink(missing_ok=True)
