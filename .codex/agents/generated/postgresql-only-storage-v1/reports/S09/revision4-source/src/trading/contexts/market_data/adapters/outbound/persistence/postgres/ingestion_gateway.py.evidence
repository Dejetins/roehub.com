"""Bounded reusable PG sessions and idempotent transaction replay (no external effects)."""

from __future__ import annotations

import ipaddress
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import tuple_row

from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)
from trading.platform.postgres_budget import (
    BudgetConnection,
    ConnectDeadlineExceeded,
    attach,
    connect_raw,
    reserve,
    validate_endpoint,
)

T = TypeVar("T")
# Installation-wide IDs, shared by every process against this database. Recovery's
# lifecycle session consumes the reconciliation reservation, even while idle.
CLASS_SLOTS = {
    "ingestion": range(0, 4),
    "interactive": range(4, 10),
    "bulk": range(10, 12),
    "recovery": range(12, 13),
    "reconcile": range(13, 14),
    "reference": range(14, 16),
}


@dataclass(frozen=True)
class ConnectionBudget:
    total: int = 48
    market: int = 16
    control: int = 24
    reserve: int = 8

    def __post_init__(self):
        if self.market != 16 or min(self.control, self.reserve) < 1:
            raise ValueError("invalid market/control/reserve budget")
        if self.market + self.control + self.reserve > self.total:
            raise ValueError("installation connection allocations exceed total")


def retryable(error: Exception) -> bool:
    state = getattr(error, "sqlstate", None)
    return (
        state in {"40001", "40P01", "55P03", "57P01", "57P02", "57P03"}
        or (isinstance(state, str) and state.startswith("08"))
        or (state is None and isinstance(error, psycopg.OperationalError))
    )


class IngestionGateway:
    """A finite local pool plus DB-wide session slots, no third-party pool dependency.

    Lifecycle owner calls close after draining. Slot admission happens before any
    transaction, and duplicate workers cannot exceed the global class allocation.
    S06 must include control/lease sessions and connection handshakes in its topology.
    """

    def __init__(
        self,
        dsn: str,
        *,
        workload: str = "ingestion",
        size: int = 1,
        max_waiters: int = 8,
        deadline_s: float = 5,
        budget: ConnectionBudget = ConnectionBudget(),
        prepare_partitions: bool = False,
    ):
        if workload not in CLASS_SLOTS or not 1 <= size <= len(CLASS_SLOTS[workload]):
            raise ValueError("invalid pool allocation")
        if max_waiters < 0 or not 0 < deadline_s <= 30:
            raise ValueError("invalid admission/deadline")
        self.budget = budget
        self.prepare_partitions = prepare_partitions
        self._months: set = set()
        self._months_lock = threading.Lock()
        self._dsn = dsn
        _endpoint_params(dsn)
        self._slots, self._deadline = CLASS_SLOTS[workload], deadline_s
        self._pool: queue.Queue[Any] = queue.Queue(size)
        for _ in range(size):
            self._pool.put(None)
        self._admission = threading.BoundedSemaphore(size + max_waiters)
        self._closed = False

    def prepare_months(self, dates):
        if not self.prepare_partitions:
            return
        months = {d.date().replace(day=1) for d in dates}
        with self._months_lock:
            for month in sorted(months - self._months):
                self.run(
                    lambda c: c.execute("SELECT market_data.ensure_partitions(%s,1)", (month,))
                )
                if len(self._months) >= 256:
                    self._months.clear()
                self._months.add(month)

    def _connect(self, stop):
        conn = _connect_until(
            make_conninfo(
                _resolve_until(self._dsn, stop),
                application_name="roehub-market-" + str(self._slots.start),
            ),
            stop,
        )
        timer = threading.Timer(max(0.001, stop - time.monotonic()), conn.close)
        timer.daemon = True
        timer.start()
        try:
            for slot in self._slots:
                row = conn.execute("SELECT pg_try_advisory_lock(563298, %s)", (slot,)).fetchone()
                if row and row[0]:
                    return conn
            raise TemporaryIngestionError("postgres_class_saturated")
        except BaseException:
            conn.close()
            raise
        finally:
            timer.cancel()
            timer.join()

    def run(self, operation: Callable[[Any], T], *, read_only: bool = False) -> T:
        if self._closed:
            raise RuntimeError("gateway closed")
        if not self._admission.acquire(blocking=False):
            raise TemporaryIngestionError("postgres_admission_full")
        conn = None
        acquired = False
        stop = time.monotonic() + self._deadline
        try:
            try:
                conn = self._pool.get(timeout=min(1, self._deadline))
                acquired = True
            except queue.Empty:
                raise TemporaryIngestionError("postgres_pool_timeout") from None
            for attempt in range(3):
                if self._closed:
                    raise RuntimeError("gateway closed")
                if time.monotonic() >= stop:
                    raise TemporaryIngestionError("postgres_transaction_timeout")
                try:
                    if conn is None or conn.closed:
                        conn = self._connect(stop)
                    remaining = stop - time.monotonic()
                    if remaining <= 0:
                        raise TemporaryIngestionError("postgres_transaction_timeout")
                    timer = threading.Timer(remaining, conn.close)
                    timer.daemon = True
                    timer.start()
                    try:
                        with conn.transaction():
                            if read_only:
                                conn.execute(
                                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                                )
                            conn.execute(
                                "SELECT set_config('statement_timeout',%s,true), "
                                "set_config('lock_timeout','250ms',true), "
                                "set_config('idle_in_transaction_session_timeout','5s',true), "
                                "set_config('timezone','UTC',true), "
                                "set_config('synchronous_commit','on',true)",
                                (str(max(1, int(remaining * 1000))),),
                            )
                            result = operation(conn)
                        # Returning from transaction is the only successful receipt.
                        return result
                    finally:
                        timer.cancel()
                        timer.join()
                except Exception as exc:
                    if conn is not None:
                        conn.close()
                    conn = None
                    if not retryable(exc):
                        raise
                    if attempt == 2 or time.monotonic() + 0.05 * (attempt + 1) >= stop:
                        raise TemporaryIngestionError("postgres_retry_exhausted") from None
                    time.sleep(0.05 * (attempt + 1))
            raise AssertionError("unreachable")
        finally:
            if acquired:
                if self._closed and conn is not None:
                    conn.close()
                    conn = None
                self._pool.put_nowait(conn)
            self._admission.release()

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                break
            if conn is not None:
                conn.close()


def _endpoint_params(dsn: str) -> dict[str, str]:
    params = {k: str(v) for k, v in conninfo_to_dict(dsn).items() if v is not None}
    host = params.get("host", "")
    if not host or params.get("service") or "," in host or "," in params.get("port", ""):
        raise ValueError("gateway requires one explicit PostgreSQL endpoint")
    return params


def _resolved_dsn(dsn: str) -> str:
    """Validate prepared conninfo; preserve original host for TLS verification."""
    params = _endpoint_params(dsn)
    host, address = params["host"], params.get("hostaddr", "")
    if host.startswith("/"):
        return dsn
    if address:
        ipaddress.ip_address(address)
    else:
        ipaddress.ip_address(host)
        params["hostaddr"] = host
    return make_conninfo("", **params)


def _resolve_until(dsn: str, stop: float) -> str:
    """Kill and reap bounded DNS work on timeout; re-resolve every new connection.

    The child sees hostname/port only, never credentials. Original host/TLS settings
    remain intact. libpq tries resolved IPv4/IPv6 addresses inside the same deadline.
    """
    params = _endpoint_params(dsn)
    host = params["host"]
    if host.startswith("/"):
        return dsn
    if params.get("hostaddr"):
        return _resolved_dsn(dsn)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        remaining = stop - time.monotonic()
        if remaining <= 0:
            raise TemporaryIngestionError("postgres_dns_timeout")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import socket,json,sys,os,time,threading\n"
                    "parent=int(sys.argv[3]);stop=float(sys.argv[4])\n"
                    "def watch():\n"
                    " while os.getppid()==parent and time.monotonic()<stop: time.sleep(.05)\n"
                    " os._exit(124)\n"
                    "threading.Thread(target=watch,daemon=True).start()\n"
                    "print(json.dumps(list(dict.fromkeys(r[4][0] for r in "
                    "socket.getaddrinfo(sys.argv[1],sys.argv[2],type=socket.SOCK_STREAM)))))",
                    host,
                    params.get("port", "5432"),
                    str(os.getpid()),
                    str(stop),
                ],
                capture_output=True,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise TemporaryIngestionError("postgres_dns_timeout") from None
        if result.returncode:
            raise TemporaryIngestionError("postgres_dns_unavailable")
        addresses = json.loads(result.stdout)
        if not addresses:
            raise TemporaryIngestionError("postgres_dns_unavailable")
        for address in addresses:
            ipaddress.ip_address(address)
        params["hostaddr"] = addresses[0]
        params["host"] = host
    else:
        params["hostaddr"] = host
    return make_conninfo("", **params)


def _connect_until(dsn: str, stop: float):
    """Public libpq nonblocking polling; all waits share the transaction deadline.

    PG16 PQconnectPoll requires hostaddr to avoid blocking DNS and requires the
    caller to implement timeout (connect_timeout is ignored by this API).
    """
    dsn = validate_endpoint(dsn)
    reservation = reserve("market", deadline=stop, dsn=dsn)
    try:
        pgconn = connect_raw(dsn, reservation, stop)
    except ConnectDeadlineExceeded:
        # Transport cleanup already ran; preserve the failed phase instead of
        # reporting retries exhausted when the handshake consumed the deadline.
        raise TemporaryIngestionError("postgres_connect_timeout") from None
    try:
        conn = BudgetConnection(pgconn, row_factory=tuple_row)
        conn.autocommit = True
        timer = threading.Timer(max(0.001, stop - time.monotonic()), conn.close)
        timer.daemon = True
        timer.start()
        try:
            attach(conn, reservation)
        finally:
            timer.cancel()
            timer.join()
        return conn
    except BaseException:
        pgconn.finish()
        if reservation is not None:
            reservation.finish_transport()
        raise
