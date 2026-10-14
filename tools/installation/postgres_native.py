"""Count a single native PostgreSQL session using a loopback byte relay.

Run pg_dump/pg_restore/psql on the trusted operator host. The relay verifies the
remote TLS certificate against the configured CA and original hostname; the
native utility uses a local plaintext connection. Parallel jobs, reconnect and
shell commands are not supported. Unknown outcomes use operator budget recovery.
"""

from __future__ import annotations

import argparse
import os
import select
import socket
import ssl
import stat
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from psycopg.conninfo import conninfo_to_dict

from trading.platform.postgres_budget import reserve, validate_endpoint
from trading.platform.postgres_dns import _resolve_until


def native_arguments(utility: str, arguments: list[str]) -> None:
    """Closed grammar: getopt abbreviations/clusters must not override endpoint."""
    if utility == "psql":
        if len(arguments) != 2 or arguments[0] not in ("-c", "--command") or "\\" in arguments[1]:
            raise ValueError("psql guard accepts one SQL -c command without backslash commands")
        return
    allowed = {"--no-owner", "--no-privileges", "--verbose"}
    allowed |= (
        {"--format=custom", "--format=plain", "--data-only", "--schema-only"}
        if utility == "pg_dump"
        else {"--exit-on-error", "--clean", "--if-exists", "--single-transaction"}
    )
    archive = False
    for argument in arguments:
        if argument in allowed:
            continue
        if utility == "pg_restore" and not archive and argument.startswith("/"):
            archive = True
            continue
        raise ValueError("unsupported native argument; use complete allowlisted options")


def original_password(params: dict[str, str]) -> str | None:
    """Resolve pgpass against the remote endpoint before choosing a relay port."""
    if "password" in params:
        return params["password"]
    if "PGPASSWORD" in os.environ:
        return os.environ["PGPASSWORD"]
    path = Path(params.get("passfile") or os.environ.get("PGPASSFILE") or Path.home() / ".pgpass")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 65536:
            raise ValueError("native pgpass requires a protected regular file of at most 64 KiB")
        content = os.read(fd, 65537).decode()
    finally:
        os.close(fd)
    expected = [params[key] for key in ("host", "port", "dbname", "user")]
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [""]
        escaped = False
        for char in line:
            if escaped:
                fields[-1] += char
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == ":":
                fields.append("")
            else:
                fields[-1] += char
        if len(fields) == 5 and all(a == "*" or a == b for a, b in zip(fields[:4], expected)):
            return fields[4]
    return None


def run_native(dsn: str, runner: Callable[[dict[str, str]], Any], *, timeout: float = 300) -> Any:
    if timeout <= 0 or timeout > 3600:
        raise ValueError("native timeout must be in (0,3600]")
    endpoint = _resolve_until(validate_endpoint(dsn), time.monotonic() + 3)
    params = {
        key: str(value) for key, value in conninfo_to_dict(endpoint).items() if value is not None
    }
    if not params.get("hostaddr") or params["host"].startswith("/"):
        raise ValueError("native guard requires one TCP PostgreSQL endpoint")
    if not params.get("dbname") or not params.get("user"):
        raise ValueError("native guard requires explicit database and user")
    password = original_password(params)
    mode = params.get("sslmode", "disable")
    if mode not in {"disable", "verify-full"}:
        raise ValueError("native guard supports sslmode=disable or verify-full")
    if params.get("channel_binding", "prefer") == "require":
        raise ValueError("native loopback guard does not support required channel binding")
    tls = None
    if mode == "verify-full":
        if not params.get("sslrootcert"):
            raise ValueError("native TLS requires an explicit sslrootcert")
        # Preserve the installation's verify-full contract. Extra libpq TLS
        # controls are rejected rather than silently ignored by Python's SSL.
        unsupported = [
            key for key in params if key.startswith("ssl") and key not in {"sslmode", "sslrootcert"}
        ]
        if unsupported:
            raise ValueError("native TLS does not support additional libpq SSL options")
        tls = ssl.create_default_context(cafile=params["sslrootcert"])
    reservation = reserve("admin", deadline=time.monotonic() + 2, dsn=endpoint)
    if reservation is None:
        raise ValueError("native guard requires the installation registry")
    upstream = None
    listener = None
    local = None
    worker = None
    stop = threading.Event()
    failure = []
    try:
        reservation.endpoint(endpoint)
        address = params["hostaddr"]
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        upstream = socket.socket(family, socket.SOCK_STREAM)
        reservation.peer = socket.socket(fileno=os.dup(upstream.fileno()))
        upstream.settimeout(3)
        upstream.connect((address, int(params["port"])))
        if tls is not None:
            upstream.sendall(struct.pack("!II", 8, 80877103))
            if upstream.recv(1) != b"S":
                raise RuntimeError("PostgreSQL TLS negotiation refused")
            upstream = tls.wrap_socket(upstream, server_hostname=params["host"])
        upstream.setblocking(False)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)
        native = dict(params)
        native.pop("passfile", None)
        if password is not None:
            native["password"] = password
        native["hostaddr"] = "127.0.0.1"
        native["port"] = str(listener.getsockname()[1])
        native["application_name"] = reservation.record["token"]
        native["sslmode"] = "disable"
        native.pop("sslrootcert", None)
        native.pop("connect_timeout", None)
        native["connect_timeout"] = "3"
        deadline = time.monotonic() + timeout

        def relay():
            nonlocal local
            try:
                local, _ = listener.accept()
                listener.close()  # Exactly one downstream session; reconnect is refused.
                local.setblocking(False)
                peers = [local, upstream]
                buffers = {local: bytearray(), upstream: bytearray()}
                while not stop.is_set() and time.monotonic() < deadline:
                    reads = [
                        p for p in peers if len(buffers[upstream if p is local else local]) < 65536
                    ]
                    writes = [p for p in peers if buffers[p]]
                    pending = isinstance(upstream, ssl.SSLSocket) and upstream.pending() > 0
                    readable, writable, _ = select.select(reads, writes, [], 0 if pending else 0.05)
                    if pending and upstream in reads and upstream not in readable:
                        readable.append(upstream)
                    for peer in writable:
                        try:
                            sent = peer.send(buffers[peer])
                        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                            continue
                        del buffers[peer][:sent]
                    for peer in readable:
                        try:
                            data = peer.recv(65536)
                        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                            continue
                        if not data:
                            peers.remove(peer)
                            if peer is upstream and not buffers[local]:
                                return
                            # Native client closed. Flush its final Terminate bytes,
                            # then drain server completion without local TCP shutdown.
                            if not buffers[upstream]:
                                return
                        else:
                            buffers[upstream if peer is local else local].extend(data)
                    if local not in peers and not buffers[upstream]:
                        return
                    if upstream not in peers and not buffers[local]:
                        return
                if not stop.is_set():
                    raise TimeoutError("native relay deadline")
            except BaseException as error:
                failure.append(type(error).__name__)

        worker = threading.Thread(target=relay, daemon=True)
        worker.start()
        result = runner(native)
        # Runner must itself be bounded. CLI subprocess has the same deadline.
        worker.join(max(0, deadline - time.monotonic()))
        if worker.is_alive():
            raise TimeoutError("native relay did not finish")
        if failure:
            raise RuntimeError("native relay transport failed")
        return result
    finally:
        stop.set()
        if worker is not None:
            worker.join(3.1)
        if local is not None:
            local.close()
        if listener is not None:
            listener.close()
        if upstream is not None:
            upstream.close()
        reservation.finish_transport()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", default="ROEHUB_STORAGE_POSTGRES_DSN")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("utility", choices=("pg_dump", "pg_restore", "psql"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        native_arguments(args.utility, args.arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    from psycopg.conninfo import make_conninfo

    def runner(options):
        environment = dict(os.environ)
        for key in (
            "PGSERVICE",
            "PGSERVICEFILE",
            "PGHOST",
            "PGHOSTADDR",
            "PGPORT",
            "PGDATABASE",
            "PGPASSFILE",
        ):
            environment.pop(key, None)
        connection_options = dict(options)
        if "password" in connection_options:
            environment["PGPASSWORD"] = connection_options.pop("password")
        command = [
            args.utility,
            "--no-password",
            "--dbname",
            make_conninfo("", **connection_options),
            *args.arguments,
        ]
        if args.utility == "psql":
            command.insert(1, "--no-psqlrc")
        return subprocess.run(
            command, env=environment, timeout=args.timeout, check=False
        ).returncode

    raise SystemExit(run_native(os.environ[args.dsn_env], runner, timeout=args.timeout))


if __name__ == "__main__":
    main()
