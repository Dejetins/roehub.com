"""Installation-wide pre-handshake PostgreSQL reservations on a shared POSIX volume.

The registry is deliberately fail-closed: a dead process does not free a reservation.
Only raw peer EOF on the original transport frees a reservation.
Unknown authentication outcomes stay quarantined until recovery is proven.
No credentials or SQL parameters are persisted. All replicas must share this volume.
"""

from __future__ import annotations

import fcntl
import json
import os
import select
import socket
import stat
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg import pq
from psycopg.adapt import AdaptersMap
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import tuple_row

LIMITS = {"market": 16, "control": 24, "admin": 8}


class BudgetExhausted(psycopg.OperationalError):
    """No physical connection reservation became available before the deadline."""


class ConnectDeadlineExceeded(psycopg.OperationalError):
    """The absolute deadline expired during the physical connection handshake."""


class Reservation:
    def __init__(self, fd: int, token: str, *, dsn: str | None = None) -> None:
        self._lock = threading.RLock()
        self.fd = fd
        self.peer: socket.socket | None = None
        self.record: dict[str, Any] = {"token": token, "state": "connecting"}
        if dsn is None:
            self.save()
        else:
            self._prepare_endpoint(dsn)

    def save(self) -> None:
        with self._lock:
            if self.fd < 0:
                raise BudgetExhausted("postgres_reservation_already_closed")
            raw = json.dumps(self.record, sort_keys=True).encode()
            os.pwrite(self.fd, raw, 0)
            os.ftruncate(self.fd, len(raw))
            os.fsync(self.fd)

    def connected(self, connection: psycopg.Connection[Any]) -> None:
        # The connection is already counted. pg_stat_activity visibility of self
        # establishes identity, and avoids freeing a reused PID after a crash.
        row = connection.execute(
            "SELECT backend_start::text, pg_postmaster_start_time()::text, "
            "(pg_control_system()).system_identifier::text "
            "FROM pg_stat_activity WHERE pid=pg_backend_pid()"
        ).fetchone()
        values = list(row.values()) if isinstance(row, dict) else row
        if values is None:
            raise BudgetExhausted("postgres_backend_identity_unavailable")
        bind_authority(values[2], values[1])
        self.record.update(
            state="connected",
            pid=connection.info.backend_pid,
            start=values[0],
            epoch=values[1],
            cluster=values[2],
        )
        self.save()

    def observe_socket(self, pgconn: Any) -> None:
        # Capture after connect_start, before subsequent polls can close the socket.
        if pgconn.socket >= 0:
            self.peer = socket.socket(fileno=os.dup(pgconn.socket))

    def endpoint(self, dsn: str) -> None:
        """Durably associate every attempt with a destination before socket I/O."""
        if self.peer is not None:
            raise BudgetExhausted("postgres_endpoint_already_prepared")
        if "endpoint" in self.record:
            params = conninfo_to_dict(dsn)
            if self.record["endpoint"] != {
                "address": params["hostaddr"],
                "port": int(str(params["port"])),
            }:
                raise BudgetExhausted("postgres_endpoint_already_prepared")
            return
        try:
            self._prepare_endpoint(dsn)
        except BaseException:
            # No socket exists yet: a rejected destination is not an unknown
            # server-side outcome and must not poison operator recovery.
            os.ftruncate(self.fd, 0)
            os.fsync(self.fd)
            self.abandon()
            raise

    def _prepare_endpoint(self, dsn: str) -> None:
        params = conninfo_to_dict(dsn)
        target = {"address": params["hostaddr"], "port": int(str(params["port"]))}
        root = Path(os.environ["ROEHUB_PG_BUDGET_ROOT"])
        try:
            fd = os.open(root / "endpoint.json", os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            # Bootstrap before operator enrollment. Recovery subsequently refuses
            # every unknown/malformed/foreign target rather than clearing it.
            pass
        else:
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                    raise BudgetExhausted("postgres_endpoint_binding_unprotected")
                if json.loads(os.pread(fd, 1024, 0)) != target:
                    raise BudgetExhausted("postgres_endpoint_binding_mismatch")
            finally:
                os.close(fd)
        self.record["endpoint"] = target
        self.save()

    def finish_transport(self) -> None:
        """Release only a Linux TCP FIN witness after PQfinish, before local shutdown.

        Raw EOF alone is ambiguous after libpq consumes RST. TCP_CLOSE_WAIT proves
        the peer sent FIN while our write half remained open. No local shutdown
        is performed. Unknown errors/timeouts and unsupported platforms quarantine.
        """
        with self._lock:
            if self.fd < 0:
                return
            try:
                if self.peer is None or sys.platform != "linux":
                    return
                stop = time.monotonic() + 1
                while time.monotonic() < stop:
                    remaining = max(0, stop - time.monotonic())
                    if not select.select([self.peer], [], [], remaining)[0]:
                        break
                    try:
                        data = self.peer.recv(65536, socket.MSG_DONTWAIT)
                    except BlockingIOError:
                        continue
                    except OSError:
                        break
                    if not data:
                        try:
                            state = self.peer.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)[
                                0
                            ]
                        except OSError:
                            break
                        if state == 8:  # Linux TCP_CLOSE_WAIT: received FIN, local write open.
                            os.ftruncate(self.fd, 0)
                            os.fsync(self.fd)
                        break
            finally:
                if self.peer is not None:
                    self.peer.close()
                    self.peer = None
                self.abandon()

    def abandon(self) -> None:
        with self._lock:
            if self.fd < 0:
                return
            os.close(self.fd)
            self.fd = -1

    def release_unstarted(self) -> None:
        """Only a caller which has not called connect_start/socket may use this."""
        with self._lock:
            if self.peer is not None:
                raise RuntimeError("cannot release an attempted transport")
            if self.fd >= 0:
                os.ftruncate(self.fd, 0)
                os.fsync(self.fd)
                self.abandon()


def bind_authority(cluster: str, epoch: str) -> None:
    """Pin the registry to one physical cluster; an offline reset retains this binding."""
    root = Path(os.environ["ROEHUB_PG_BUDGET_ROOT"])
    fd = os.open(root / "authority.json", os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.pread(fd, 256, 0)
        if raw:
            authority = json.loads(raw)
            if authority != {"cluster": cluster, "epoch": epoch}:
                raise BudgetExhausted("postgres_budget_authority_mismatch")
        else:
            os.pwrite(fd, json.dumps({"cluster": cluster, "epoch": epoch}).encode(), 0)
            os.fsync(fd)
    finally:
        os.close(fd)


def reserve(kind: str, *, deadline: float, dsn: str | None = None) -> Reservation | None:
    if kind not in LIMITS:
        raise ValueError("unknown PostgreSQL connection class")
    raw_root = os.environ.get("ROEHUB_PG_BUDGET_ROOT", "")
    if not raw_root:
        if os.environ.get("ROEHUB_PG_BUDGET_REQUIRED") == "1":
            raise BudgetExhausted("postgres_budget_configuration_missing")
        return None  # Explicit non-installation development/test composition.
    root = Path(raw_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BudgetExhausted("postgres_budget_volume_unavailable")
    root_stat = root.stat()
    if root_stat.st_uid != 0 or root_stat.st_mode & 0o022:
        raise BudgetExhausted("postgres_budget_directory_not_protected")
    authority_path = root / "authority.json"
    if authority_path.stat().st_size:
        authority = json.loads(authority_path.read_text())
        if authority.get("recovery_pending"):
            raise BudgetExhausted("postgres_operator_recovery_pending")
    while True:
        for index in range(LIMITS[kind]):
            fd = os.open(root / f"{kind}-{index}.json", os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise BudgetExhausted("postgres_budget_invalid_inode")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.fstat(fd).st_size:
                    continue
                if authority_path.stat().st_size:
                    authority = json.loads(authority_path.read_text())
                    if authority.get("recovery_pending"):
                        raise BudgetExhausted("postgres_operator_recovery_pending")
                reservation = Reservation(fd, f"rh-{kind}-{uuid.uuid4().hex}", dsn=dsn)
                fd = -1
                return reservation
            except BlockingIOError:
                pass
            finally:
                if fd >= 0:
                    os.close(fd)
        if time.monotonic() >= deadline:
            raise BudgetExhausted("postgres_physical_budget_exhausted")
        time.sleep(min(0.02, max(0, deadline - time.monotonic())))


class BudgetConnection(psycopg.Connection):
    _reservation: Reservation | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._budget_close_lock = threading.Lock()

    def close(self) -> None:
        with self._budget_close_lock:
            try:
                if self._reservation is not None and self.lock.acquire(blocking=False):
                    try:
                        finish_idle_tls(self.pgconn, self._reservation)
                    finally:
                        self.lock.release()
                super().close()
            finally:
                if self._reservation is not None:
                    held = self._reservation
                    self._reservation = None
                    held.finish_transport()

    def cancel_safe(self, *, timeout: float = 30.0) -> None:
        self.cancel()

    def cancel(self) -> None:
        # libpq cancellation opens an additional, otherwise uncounted socket.
        raise psycopg.NotSupportedError(
            "use statement_timeout or counted administrative termination"
        )


def finish_idle_tls(pgconn: Any, reservation: Reservation) -> None:
    """Ask an idle TLS backend to exit before libpq sends its TLS close_notify.

    PostgreSQL may reset a socket with unread client close_notify after Terminate.
    Keep libpq open and drain the server's FIN first. This rolls back exactly the
    uncommitted work that close would discard. Busy/broken sessions fail closed.
    """
    if (
        pgconn.pipeline_status != pq.PipelineStatus.OFF
        or not pgconn.ssl_in_use
        or pgconn.transaction_status
        not in (
            pq.TransactionStatus.IDLE,
            pq.TransactionStatus.INTRANS,
            pq.TransactionStatus.INERROR,
        )
    ):
        return
    try:
        pgconn.send_query(
            b"ROLLBACK; SELECT pg_catalog.pg_terminate_backend(pg_catalog.pg_backend_pid())"
        )
        deadline = time.monotonic() + 1
        while pgconn.flush():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [pgconn.socket], [], remaining)[1]:
                return
        reservation.finish_transport()
    except (psycopg.Error, OSError):
        pass


def connect_raw(dsn: str, reservation: Reservation | None, stop: float) -> Any:
    """One nonblocking direct endpoint, with an owner-held transport witness."""
    pgconn = None
    try:
        if reservation is not None:
            reservation.endpoint(dsn)
        pgconn = pq.PGconn.connect_start(prepared_dsn(dsn, reservation).encode())
        if reservation is not None:
            reservation.observe_socket(pgconn)
        state = pq.PollingStatus.WRITING
        while state != pq.PollingStatus.OK:
            if pgconn.status == pq.ConnStatus.BAD or state == pq.PollingStatus.FAILED:
                raise psycopg.OperationalError("PostgreSQL connection failed")
            remaining = stop - time.monotonic()
            if remaining <= 0:
                raise ConnectDeadlineExceeded("PostgreSQL connection timed out")
            reading = [pgconn.socket] if state == pq.PollingStatus.READING else []
            writing = [pgconn.socket] if state == pq.PollingStatus.WRITING else []
            if reading or writing:
                ready = select.select(reading, writing, [], remaining)
                if not (ready[0] or ready[1]):
                    raise ConnectDeadlineExceeded("PostgreSQL connection timed out")
            state = pq.PollingStatus(pgconn.connect_poll())
        pgconn.nonblocking = 1
        return pgconn
    except BaseException:
        if pgconn is not None:
            pgconn.finish()
        if reservation is not None:
            reservation.finish_transport()
        raise


def configure_connection(connection: Any, options: dict[str, Any]) -> None:
    connection._autocommit = bool(options.get("autocommit", False))
    if options.get("row_factory"):
        connection.row_factory = options["row_factory"]
    if options.get("cursor_factory"):
        connection.cursor_factory = options["cursor_factory"]
    if options.get("context"):
        connection._adapters = AdaptersMap(options["context"].adapters)
    connection.prepare_threshold = options.get("prepare_threshold", 5)


def connect(
    conninfo: str = "", *, budget_class: str = "control", reap_closed: bool = False, **kwargs: Any
) -> psycopg.Connection:
    if os.environ.get("ROEHUB_PG_BUDGET_ROOT"):
        connection_options = {
            "autocommit",
            "row_factory",
            "cursor_factory",
            "context",
            "prepare_threshold",
        }
        transport = {key: kwargs.pop(key) for key in tuple(kwargs) if key not in connection_options}
        conninfo = validate_endpoint(make_conninfo(conninfo, **transport))
        from trading.platform.postgres_dns import _resolve_until

        conninfo = _resolve_until(conninfo, time.monotonic() + 3)
    reservation = reserve(budget_class, deadline=time.monotonic() + 2, dsn=conninfo)
    if reservation is None:
        return psycopg.connect(conninfo, **kwargs)
    connection = None
    try:
        pgconn = connect_raw(conninfo, reservation, time.monotonic() + 3)
        connection = BudgetConnection(pgconn)
        connection._reservation = reservation
        configure_connection(connection, kwargs)
        was_autocommit = connection.autocommit
        connection.autocommit = True
        timer = threading.Timer(3, connection.close)
        timer.daemon = True
        timer.start()
        try:
            reservation.connected(connection)
            if reap_closed:
                reap(connection)
        finally:
            timer.cancel()
            timer.join()
        connection.autocommit = was_autocommit
        return connection
    except BaseException:
        if connection is not None:
            connection.close()
        reservation.abandon()
        raise


def attach(connection: BudgetConnection, reservation: Reservation | None) -> None:
    connection._reservation = reservation
    if reservation is not None:
        reservation.connected(connection)


def prepared_dsn(dsn: str, reservation: Reservation | None) -> str:
    return make_conninfo(dsn, application_name=reservation.record["token"]) if reservation else dsn


def reservation_counts() -> dict[str, dict[str, int]]:
    """Approximate bounded operational counts; no record contents or identifiers."""
    raw = os.environ.get("ROEHUB_PG_BUDGET_ROOT")
    if not raw:
        return {}
    result = {}
    for kind, limit in LIMITS.items():
        counts = {"active": 0, "quarantined": 0, "limit": limit}
        for index in range(limit):
            fd = os.open(Path(raw) / f"{kind}-{index}.json", os.O_RDONLY | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    counts["active"] += 1
                else:
                    counts["quarantined"] += bool(os.fstat(fd).st_size)
            finally:
                os.close(fd)
        result[kind] = counts
    return result


def reap(connection: psycopg.Connection[Any]) -> int:
    """Unknown/crashed outcomes remain quarantined.

    pg_stat_activity absence precedes physical backend exit and cannot authorize
    reuse. Transport-confirmed completions are freed by their original owner.
    Unknown outcomes require the enrolled operator stop/reset/resume procedure.
    """
    return 0


def admin_connect(conninfo: str = "", **kwargs: Any) -> psycopg.Connection:
    return connect(conninfo, budget_class="admin", reap_closed=True, **kwargs)


def validate_endpoint(dsn: str) -> str:
    """One explicit endpoint; no libpq implicit failover or encryption retry."""
    params = conninfo_to_dict(dsn)
    if (
        not params.get("host")
        or params.get("service")
        or os.environ.get("PGSERVICE")
        or os.environ.get("PGSERVICEFILE")
    ):
        raise BudgetExhausted("postgres_explicit_endpoint_required")
    if any("," in str(params.get(key, "")) for key in ("host", "hostaddr", "port")):
        raise BudgetExhausted("postgres_multihost_forbidden")
    if params.get("target_session_attrs", "any") != "any":
        raise BudgetExhausted("postgres_target_probe_forbidden")
    if params.get("sslmode", "disable") not in {"disable", "require", "verify-ca", "verify-full"}:
        raise BudgetExhausted("postgres_encryption_fallback_forbidden")
    params.setdefault("port", "5432")
    params["target_session_attrs"] = "any"
    params.setdefault("sslmode", "disable")
    params["gssencmode"] = "disable"
    return make_conninfo("", **params)


class AsyncBudgetConnection(psycopg.AsyncConnection):
    _reservation: Reservation | None = None

    async def close(self) -> None:
        try:
            if self._reservation is not None and not self.lock.locked():
                # No await in this bounded section: another task cannot acquire
                # the async connection lock between this check and finalization.
                finish_idle_tls(self.pgconn, self._reservation)
            await super().close()
        finally:
            if self._reservation is not None:
                self._reservation.finish_transport()
                self._reservation = None

    def cancel(self) -> None:
        raise psycopg.NotSupportedError("use statement_timeout or close")

    async def cancel_safe(self, *, timeout: float = 30.0) -> None:
        await self.close()

    async def _try_cancel(self, *, timeout: float = 5.0) -> None:
        await self.close()


async def async_connect(conninfo: str, **kwargs: Any) -> psycopg.AsyncConnection[Any]:
    """One counted lifecycle connection for async journal/control consumers."""
    import asyncio

    from trading.platform.postgres_dns import _resolve_until

    if not os.environ.get("ROEHUB_PG_BUDGET_ROOT"):
        reserve("control", deadline=time.monotonic())  # Enforce REQUIRED before network I/O.
        return await psycopg.AsyncConnection.connect(conninfo, **kwargs)
    options = {"autocommit", "row_factory", "cursor_factory", "context", "prepare_threshold"}
    transport = {key: kwargs.pop(key) for key in tuple(kwargs) if key not in options}
    dsn = validate_endpoint(make_conninfo(conninfo, **transport))

    # A cancelled caller waits only for the bounded DNS/reservation task to finish.
    def prepare():
        resolved = _resolve_until(dsn, time.monotonic() + 3)
        return resolved, reserve("control", deadline=time.monotonic() + 2, dsn=resolved)

    def release_prepared(value):
        _, held = value
        if held is not None:
            held.release_unstarted()

    resolved, reservation = await _owned_thread_call(prepare, release_prepared)
    assert reservation is not None

    def close_opened(pgconn):
        try:
            finish_idle_tls(pgconn, reservation)
        finally:
            pgconn.finish()
            reservation.finish_transport()

    pgconn = await _owned_thread_call(
        lambda: connect_raw(resolved, reservation, time.monotonic() + 3), close_opened
    )
    connection = None
    try:
        async with asyncio.timeout(6):
            connection = AsyncBudgetConnection(pgconn, row_factory=tuple_row)
            connection._reservation = reservation
            configure_connection(connection, kwargs)
            previous_autocommit = connection.autocommit
            await connection.set_autocommit(True)
            row = await (
                await connection.execute(
                    "SELECT backend_start::text, pg_postmaster_start_time()::text, "
                    "(pg_control_system()).system_identifier::text "
                    "FROM pg_stat_activity WHERE pid=pg_backend_pid()"
                )
            ).fetchone()
            values = list(row.values()) if isinstance(row, dict) else row
            if values is None:
                raise BudgetExhausted("postgres_backend_identity_unavailable")
            bind_authority(values[2], values[1])
            reservation.record.update(
                state="connected",
                pid=connection.info.backend_pid,
                start=values[0],
                epoch=values[1],
                cluster=values[2],
            )
            reservation.save()
            await connection.set_autocommit(previous_autocommit)
        return connection
    except BaseException:
        if connection is not None:
            await connection.close()
        reservation.abandon()
        raise


async def _owned_thread_call(operation: Any, cleanup: Any) -> Any:
    """Transfer a bounded thread result exactly once, even under repeated cancellation."""
    import asyncio

    lock = threading.Lock()
    state: dict[str, Any] = {"cancelled": False}

    def work():
        value = operation()
        with lock:
            cancelled = state["cancelled"]
            if not cancelled:
                state["value"] = value
        if cancelled:
            cleanup(value)
        return value

    task = asyncio.create_task(asyncio.to_thread(work))
    # Consume errors if the caller is already cancelled. Cleanup ownership lives
    # in the thread/state, not in a second await that another cancel can interrupt.
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        value = await asyncio.shield(task)
    except BaseException:
        with lock:
            state["cancelled"] = True
            value = state.pop("value", None)
        if value is not None:
            cleanup(value)
        raise
    with lock:
        state.pop("value", None)
    return value
