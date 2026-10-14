"""Operator-only recovery using an enrolled, dedicated remote Docker PostgreSQL.

The Docker socket may be a trusted SSH-forwarded remote daemon socket. It is an
administrative channel, never mounted into ordinary application containers.
No server data volume is shared with clients. Enrollment proves the counted SQL
connection belongs to the selected container by its backend-owned TCP socket.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import http.client
import ipaddress
import json
import os
import socket
import stat
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from psycopg.conninfo import make_conninfo

from trading.platform.postgres_budget import (
    LIMITS,
    BudgetConnection,
    Reservation,
    connect,
    connect_raw,
    validate_endpoint,
)
from trading.platform.postgres_dns import _resolve_until


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path: str) -> None:
        super().__init__("localhost", timeout=60)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class EngineHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"administrative API failed: HTTP {status}")


class Engine:
    def __init__(self, path: str) -> None:
        self.path = path
        self.deadline: float | None = None

    def call(self, method: str, path: str, body: Any = None, *, raw: bool = False) -> Any:
        connection = UnixHTTP(self.path)
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("administrative operation deadline")
            connection.timeout = min(60, remaining)
        try:
            connection.request(
                method,
                "/v1.47" + path,
                body=None if body is None else json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            data = response.read()
            if response.status not in (200, 201, 204, 304):
                raise EngineHTTPError(response.status)
            return data if raw else (json.loads(data) if data else None)
        finally:
            connection.close()

    def inspect(self, container: str) -> dict[str, Any]:
        return self.call("GET", f"/containers/{quote(container, safe='')}/json")

    def execute(self, container: str, command: list[str], *, user: str = "0") -> str:
        execution = self.call(
            "POST",
            f"/containers/{container}/exec",
            {
                "Cmd": command,
                "User": user,
                "AttachStdout": True,
                "AttachStderr": True,
            },
        )["Id"]
        data = self.call(
            "POST", f"/exec/{execution}/start", {"Detach": False, "Tty": False}, raw=True
        )
        chunks = []
        while data:
            if len(data) < 8:
                raise RuntimeError("invalid administrative output framing")
            length = struct.unpack(">I", data[4:8])[0]
            chunks.append(data[8 : 8 + length])
            data = data[8 + length :]
        status = self.call("GET", f"/exec/{execution}/json")
        if status["Running"] or status["ExitCode"] != 0:
            raise RuntimeError("administrative server inspection failed")
        return b"".join(chunks).decode()


def data_identity(record: dict[str, Any]) -> list[dict[str, Any]]:
    pgdata = next((x[7:] for x in record["Config"]["Env"] if x.startswith("PGDATA=")), None)
    if not pgdata:
        raise RuntimeError("explicit PGDATA required")
    mounts = [
        {key: item.get(key) for key in ("Type", "Name", "Source", "Destination", "RW")}
        for item in record["Mounts"]
        if pgdata == item["Destination"] or pgdata.startswith(item["Destination"].rstrip("/") + "/")
    ]
    if not mounts:
        raise RuntimeError("persistent PostgreSQL data mount required")
    return mounts


def server_cluster(engine: Engine, container: str) -> str:
    raw = engine.execute(container, ["sh", "-ec", 'LC_ALL=C pg_controldata "$PGDATA"'])
    for line in raw.splitlines():
        if line.startswith("Database system identifier:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("server cluster identity unavailable")


def address(value: str) -> tuple[str, int]:
    host, port = value.split(":")
    raw = bytes.fromhex(host)
    raw = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
    return str(ipaddress.ip_address(raw)), int(port, 16)


def prove_sql_target(engine: Engine, container: str, row: tuple[Any, ...]) -> None:
    _, local, local_port, peer, peer_port, pid = row
    if local is None or peer is None or type(pid) is not int:
        raise RuntimeError("enrollment requires a live TCP PostgreSQL connection")
    local = str(ipaddress.ip_interface(local).ip)
    peer = str(ipaddress.ip_interface(peer).ip)
    # The SQL connection remains open while its actual backend FD is inspected.
    backend_user = engine.execute(container, ["stat", "-c", "%u", f"/proc/{pid}"]).strip()
    if not backend_user.isdigit():
        raise RuntimeError("backend process owner unavailable")
    links = engine.execute(
        container,
        ["sh", "-ec", 'for f in /proc/"$1"/fd/*; do readlink "$f" || true; done', "sh", str(pid)],
        user=backend_user,
    )
    inodes = {x[8:-1] for x in links.splitlines() if x.startswith("socket:[") and x.endswith("]")}
    network = engine.execute(container, ["cat", f"/proc/{pid}/net/tcp", f"/proc/{pid}/net/tcp6"])
    for line in network.splitlines():
        cells = line.split()
        if len(cells) > 9 and cells[9] in inodes:
            if address(cells[1]) == (local, local_port) and address(cells[2]) == (peer, peer_port):
                return
    raise RuntimeError("selected administrative target does not own the SQL connection")


def protected_root(root: Path) -> None:
    if os.geteuid() != 0 or not root.is_absolute() or root.is_symlink():
        raise RuntimeError("recovery requires root and an absolute protected registry")
    metadata = root.stat()
    if metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise RuntimeError("registry directory is not protected")


@contextlib.contextmanager
def registry_lock(root: Path):
    handles = []
    try:
        fd = os.open(root / "recovery.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        handles.append(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for kind, limit in LIMITS.items():
            for index in range(limit):
                fd = os.open(root / f"{kind}-{index}.json", os.O_RDWR | os.O_NOFOLLOW)
                handles.append(fd)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise RuntimeError("invalid registry inode")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handles[1:]
    finally:
        for fd in reversed(handles):
            os.close(fd)


def enroll(root: Path, engine: Engine, container: str, dsn: str) -> dict[str, Any]:
    protected_root(root)
    engine.deadline = time.monotonic() + 120
    if os.environ.get("ROEHUB_PG_BUDGET_ROOT") != str(root):
        raise RuntimeError("counted SQL registry must equal recovery registry")
    target = engine.inspect(container)
    if target["HostConfig"].get("PidMode") not in ("", "private"):
        raise RuntimeError("private PostgreSQL PID namespace required")
    container = target["Id"]
    if (target["Config"].get("Labels") or {}).get("com.docker.swarm.service.id"):
        raise RuntimeError("supervised server requires its own maintenance adapter")
    with connect(dsn, budget_class="admin", autocommit=True) as connection:
        row = connection.execute(
            "SELECT (pg_control_system()).system_identifier::text, "
            "inet_server_addr()::text,inet_server_port(),inet_client_addr()::text, "
            "inet_client_port(),pg_backend_pid()"
        ).fetchone()
        if row is None:
            raise RuntimeError("SQL identity unavailable")
        prove_sql_target(engine, container, row)
        if server_cluster(engine, container) != row[0]:
            raise RuntimeError("cluster mismatch")
        epoch_row = connection.execute("SELECT pg_postmaster_start_time()::text").fetchone()
        if epoch_row is None:
            raise RuntimeError("SQL epoch unavailable")
        assert isinstance(connection, BudgetConnection)
        assert connection._reservation is not None
        pin = {
            "schema": "io.roehub.pg-recovery-authority/v1",
            "daemon": engine.call("GET", "/info")["ID"],
            "container": container,
            "epoch": epoch_row[0],
            "data_mounts": data_identity(target),
            "cluster": row[0],
            "endpoint": connection._reservation.record["endpoint"],
        }
        authority = json.loads((root / "authority.json").read_text())
        if authority != {"cluster": pin["cluster"], "epoch": pin["epoch"]}:
            raise RuntimeError("registry authority does not match enrolled SQL server")
    with registry_lock(root) as handles:
        validate_record_endpoints(handles, pin)
        # An existing enrollment must be explicitly retired by the installation owner.
        fd = os.open(
            root / "recovery-authority.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.write(fd, json.dumps(pin, sort_keys=True).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_metadata(root, "endpoint.json", pin["endpoint"], owner=(0, 0), mode=0o644)
    return {"enrolled": True, "sql_socket_target_verified": True}


def validate_record_endpoints(handles: list[int], pin: dict[str, Any]) -> None:
    if "endpoint" not in pin:
        raise RuntimeError("recovery authority requires endpoint enrollment")
    for fd in handles:
        raw = os.pread(fd, 4096, 0)
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise RuntimeError("unverifiable reservation endpoint") from None
        if not isinstance(record, dict) or record.get("endpoint") != pin["endpoint"]:
            raise RuntimeError("foreign or unknown reservation endpoint")


def recover(root: Path, engine: Engine, container: str) -> dict[str, Any]:
    protected_root(root)
    engine.deadline = time.monotonic() + 120
    pin = read_pin(root)
    with registry_lock(root) as handles:
        validate_record_endpoints(handles, pin)
        target = engine.inspect(container)
        if target["HostConfig"].get("PidMode") not in ("", "private"):
            raise RuntimeError("private PostgreSQL PID namespace required")
        authority = json.loads((root / "authority.json").read_text())
        accepted_epochs = {pin["epoch"]}
        if pin.get("recovery") == "resuming":
            accepted_epochs.add(pin["previous_epoch"])
        if (
            authority.get("cluster") != pin["cluster"]
            or authority.get("epoch") not in accepted_epochs
        ):
            raise RuntimeError("runtime and recovery authority mismatch")
        for fd in handles:
            raw = os.pread(fd, 4096, 0)
            try:
                record = json.loads(raw) if raw else {}
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(record, dict) and record.get("cluster", pin["cluster"]) != pin["cluster"]:
                raise RuntimeError("foreign cluster reservation")
        if (
            pin.get("schema") != "io.roehub.pg-recovery-authority/v1"
            or engine.call("GET", "/info")["ID"] != pin["daemon"]
            or target["Id"] != pin["container"]
            or data_identity(target) != pin["data_mounts"]
        ):
            raise RuntimeError("recovery authority mismatch")
        container = target["Id"]
        if target["State"]["Running"]:
            if server_cluster(engine, container) != pin["cluster"]:
                raise RuntimeError("enrolled cluster mismatch before stop")
            if engine.execute(container, ["cat", "/proc/1/comm"]).strip() != "postgres":
                raise RuntimeError("dedicated PostgreSQL container required")
        elif (
            pin.get("recovery") not in {"stopping", "stopped"}
            or not authority.get("recovery_pending")
            or target["State"]["Restarting"]
            or target["State"]["Pid"] != 0
        ):
            raise RuntimeError("stopped target has no verified stopping intent")
        write_runtime_authority(
            root, {"cluster": pin["cluster"], "epoch": pin["epoch"], "recovery_pending": True}
        )
        write_pin(root, {**pin, "recovery": "stopping"})
        # Caller has exclusive maintenance control: other supervisors/start commands
        # must be stopped. Do not restore policy/start automatically after any failure.
        engine.call("POST", f"/containers/{container}/update", {"RestartPolicy": {"Name": "no"}})
        if engine.inspect(container)["HostConfig"]["RestartPolicy"]["Name"] != "no":
            raise RuntimeError("could not disable automatic restart")
        engine.call("POST", f"/containers/{container}/stop?t=30")
        stopped = engine.inspect(container)["State"]
        if stopped["Running"] or stopped["Restarting"] or stopped["Pid"] != 0:
            raise RuntimeError("server stop not proven")
        count = sum(bool(os.fstat(fd).st_size) for fd in handles)
        for fd in handles:
            os.ftruncate(fd, 0)
            os.fsync(fd)
        write_pin(root, {**pin, "recovery": "stopped"})
        return {
            "cleared_reservations": count,
            "server_exit_code": stopped["ExitCode"],
            "server_oom_killed": stopped["OOMKilled"],
            "server_stopped": True,
            "restart_policy": "no",
            "operator_start_required": True,
        }


def replace_recover(root: Path, engine: Engine, container: str, supervisor: str) -> dict[str, Any]:
    """Retire a Docker-deleted enrollment under an exclusive, stopped supervisor.

    This is an explicit maintenance action, never automatic name-based recovery.
    The old container must be absent, and its exact data mount, cluster and direct
    endpoint must belong to the replacement. A durable intent makes stop retryable.
    """
    protected_root(root)
    engine.deadline = time.monotonic() + 120
    pin = read_pin(root)
    with registry_lock(root) as handles:
        validate_record_endpoints(handles, pin)
        authority = json.loads((root / "authority.json").read_text())
        if authority.get("cluster") != pin["cluster"] or authority.get("epoch") != pin["epoch"]:
            raise RuntimeError("runtime and recovery authority mismatch")
        controller = engine.inspect(supervisor)
        if (
            controller["State"]["Running"]
            or controller["State"]["Restarting"]
            or controller["State"]["Pid"] != 0
            or controller["HostConfig"]["RestartPolicy"]["Name"] != "no"
        ):
            raise RuntimeError("exclusive stopped supervisor required")
        old = pin.get("replacement", {}).get("old_container", pin["container"])
        try:
            engine.inspect(old)
        except EngineHTTPError as error:
            if error.status != 404:
                raise
        else:
            raise RuntimeError("old enrolled container still exists")
        target = engine.inspect(container)
        if (
            engine.call("GET", "/info")["ID"] != pin["daemon"]
            or target["Id"] == old
            or data_identity(target) != pin["data_mounts"]
            or target["HostConfig"].get("PidMode") not in ("", "private")
            or (target["Config"].get("Labels") or {}).get("com.docker.swarm.service.id")
        ):
            raise RuntimeError("replacement authority mismatch")
        for item in engine.call("GET", "/containers/json"):
            if item["Id"] in (target["Id"], controller["Id"]):
                continue
            other = engine.inspect(item["Id"])
            # Any other running mount of the same data source is forbidden, even
            # if it is not called PostgreSQL or has no PGDATA environment.
            if any(
                m.get("Source") == old_mount["Source"]
                for m in other["Mounts"]
                for old_mount in pin["data_mounts"]
            ):
                raise RuntimeError("another running data owner exists")
        for fd in handles:
            raw = os.pread(fd, 4096, 0)
            record = json.loads(raw) if raw else {}
            if record.get("cluster", pin["cluster"]) != pin["cluster"]:
                raise RuntimeError("foreign cluster reservation")
        intent = {
            "old_container": old,
            "new_container": target["Id"],
            "supervisor": controller["Id"],
        }
        if pin.get("replacement") not in (None, intent):
            raise RuntimeError("replacement intent mismatch")
        if target["State"]["Running"]:
            addresses = {v.get("IPAddress") for v in target["NetworkSettings"]["Networks"].values()}
            if pin["endpoint"]["address"] not in addresses:
                raise RuntimeError("replacement direct endpoint mismatch")
            if server_cluster(engine, target["Id"]) != pin["cluster"]:
                raise RuntimeError("replacement cluster mismatch")
            if engine.execute(target["Id"], ["cat", "/proc/1/comm"]).strip() != "postgres":
                raise RuntimeError("dedicated PostgreSQL replacement required")
            server_user = engine.execute(target["Id"], ["stat", "-c", "%u", "/proc/1"]).strip()
            sockets = engine.execute(
                target["Id"], ["sh", "-ec", "ls -l /proc/1/fd"], user=server_user
            )
            inodes = {part.split("]", 1)[0] for part in sockets.split("socket:[")[1:]}
            listeners = engine.execute(target["Id"], ["cat", "/proc/1/net/tcp", "/proc/1/net/tcp6"])
            if not any(
                len(cells := line.split()) > 9
                and cells[3] == "0A"
                and cells[9] in inodes
                and address(cells[1])[1] == pin["endpoint"]["port"]
                for line in listeners.splitlines()
            ):
                raise RuntimeError("replacement PostgreSQL listener not proven")
        elif (
            pin.get("replacement") != intent
            or pin.get("recovery") not in {"stopping", "stopped"}
            or not authority.get("recovery_pending")
            or target["State"]["Restarting"]
            or target["State"]["Pid"] != 0
        ):
            raise RuntimeError("replacement stopped without verified intent")
        if pin.get("replacement") is None:
            atomic_metadata(root, "retired-authority-" + old + ".json", pin, owner=(0, 0))
        write_runtime_authority(
            root, {"cluster": pin["cluster"], "epoch": pin["epoch"], "recovery_pending": True}
        )
        replacement = {
            **pin,
            "container": target["Id"],
            "replacement": intent,
            "recovery": "stopping",
        }
        write_pin(root, replacement)
        engine.call("POST", f"/containers/{target['Id']}/update", {"RestartPolicy": {"Name": "no"}})
        if engine.inspect(target["Id"])["HostConfig"]["RestartPolicy"]["Name"] != "no":
            raise RuntimeError("could not disable replacement restart")
        engine.call("POST", f"/containers/{target['Id']}/stop?t=30")
        state = engine.inspect(target["Id"])["State"]
        if state["Running"] or state["Restarting"] or state["Pid"] != 0:
            raise RuntimeError("replacement stop not proven")
        count = sum(bool(os.fstat(fd).st_size) for fd in handles)
        for fd in handles:
            os.ftruncate(fd, 0)
            os.fsync(fd)
        write_pin(root, {**replacement, "recovery": "stopped"})
        return {
            "replacement_stopped": True,
            "old_container_absent": True,
            "cleared_reservations": count,
            "operator_start_required": True,
        }


def read_pin(root: Path) -> dict[str, Any]:
    fd = os.open(root / "recovery-authority.json", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_mode & 0o077
            or info.st_nlink != 1
        ):
            raise RuntimeError("recovery enrollment is not protected")
        return json.loads(os.pread(fd, 16384, 0))
    finally:
        os.close(fd)


def atomic_metadata(
    root: Path, name: str, value: dict[str, Any], *, owner: tuple[int, int], mode: int = 0o600
) -> None:
    temporary = root / (".recovery-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        raw = json.dumps(value, sort_keys=True).encode()
        os.write(fd, raw)
        os.fchown(fd, *owner)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.replace(temporary, root / name)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(fd)
        temporary.unlink(missing_ok=True)


def write_pin(root: Path, value: dict[str, Any]) -> None:
    atomic_metadata(root, "recovery-authority.json", value, owner=(0, 0))


def write_runtime_authority(root: Path, value: dict[str, Any]) -> None:
    metadata = (root / "authority.json").stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("invalid runtime authority inode")
    atomic_metadata(root, "authority.json", value, owner=(metadata.st_uid, metadata.st_gid))


def resume(root: Path, engine: Engine, container: str, dsn: str) -> dict[str, Any]:
    protected_root(root)
    engine.deadline = time.monotonic() + 120
    pin = read_pin(root)
    with registry_lock(root) as handles:
        authority = json.loads((root / "authority.json").read_text())
        target = engine.inspect(container)
        if (
            pin.get("recovery") != "stopped"
            or not authority.get("recovery_pending")
            or authority.get("cluster") != pin["cluster"]
            or authority.get("epoch") != pin["epoch"]
            or engine.call("GET", "/info")["ID"] != pin["daemon"]
            or target["Id"] != pin["container"]
            or data_identity(target) != pin["data_mounts"]
            or target["HostConfig"].get("PidMode") not in ("", "private")
            or any(os.fstat(fd).st_size for fd in handles)
        ):
            raise RuntimeError("verified stopped/reset recovery state required")
        container = target["Id"]
        engine.call("POST", f"/containers/{container}/start")
        # Wait through the control channel, without opening readiness sockets.
        stop = time.monotonic() + 30
        while True:
            try:
                engine.execute(
                    container,
                    [
                        "sh",
                        "-ec",
                        'test "$(sed -n 8p "$PGDATA/postmaster.pid" | tr -d " ")" = "ready"',
                    ],
                )
                break
            except RuntimeError:
                if time.monotonic() >= stop:
                    raise
                time.sleep(0.2)
        if server_cluster(engine, container) != pin["cluster"]:
            raise RuntimeError("restarted data identity mismatch")
        dsn = make_conninfo(dsn, options="-c statement_timeout=3000 -c lock_timeout=250")
        endpoint = _resolve_until(validate_endpoint(dsn), time.monotonic() + 3)
        reservation = Reservation(os.dup(handles[40]), "rh-admin-recovery", dsn=endpoint)
        connection = None
        watchdog = None
        try:
            pgconn = connect_raw(endpoint, reservation, time.monotonic() + 3)
            connection = BudgetConnection(pgconn)
            connection._reservation = reservation
            connection.autocommit = True
            watchdog = threading.Timer(8, connection.close)
            watchdog.daemon = True
            watchdog.start()
            row = connection.execute(
                "SELECT (pg_control_system()).system_identifier::text, "
                "inet_server_addr()::text,inet_server_port(),inet_client_addr()::text, "
                "inet_client_port(),pg_backend_pid()"
            ).fetchone()
            if row is None or row[0] != pin["cluster"]:
                raise RuntimeError("restarted SQL identity mismatch")
            prove_sql_target(engine, container, row)
            epoch_row = connection.execute("SELECT pg_postmaster_start_time()::text").fetchone()
            if epoch_row is None:
                raise RuntimeError("SQL epoch unavailable")
            epoch = epoch_row[0]
            write_pin(
                root,
                {**pin, "epoch": epoch, "previous_epoch": pin["epoch"], "recovery": "resuming"},
            )
            write_runtime_authority(root, {"cluster": pin["cluster"], "epoch": epoch})
            write_pin(
                root,
                {
                    key: value
                    for key, value in {**pin, "epoch": epoch}.items()
                    if key not in {"recovery", "previous_epoch", "replacement"}
                },
            )
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            if connection is not None:
                connection.close()
            else:
                reservation.abandon()
        return {"resumed": True, "counted_admin_identity_verified": True, "restart_policy": "no"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enroll", "recover", "resume", "replace-recover"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--docker-socket", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--supervisor", help="Exact stopped controller for replace-recover")
    parser.add_argument("--dsn-env", default="ROEHUB_STORAGE_POSTGRES_DSN")
    args = parser.parse_args()
    engine = Engine(args.docker_socket)
    try:
        if args.action == "enroll":
            result = enroll(args.registry, engine, args.container, os.environ[args.dsn_env])
        elif args.action == "recover":
            result = recover(args.registry, engine, args.container)
        elif args.action == "replace-recover":
            if not args.supervisor:
                raise RuntimeError("explicit supervisor required")
            result = replace_recover(args.registry, engine, args.container, args.supervisor)
        else:
            result = resume(args.registry, engine, args.container, os.environ[args.dsn_env])
    except Exception as error:
        # No raw administrative responses, DSNs, environment or credentials.
        detail = str(error) if type(error) is RuntimeError else type(error).__name__
        raise SystemExit(f"PostgreSQL budget recovery refused: {detail}") from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
