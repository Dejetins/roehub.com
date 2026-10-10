"""Real PG, local HTTP transport and separate-process quota/deadline proofs."""

import multiprocessing
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import LiteralString, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from trading.contexts.market_data.adapters.outbound.clients.common_http import RequestsHttpClient
from trading.contexts.market_data.adapters.outbound.persistence.postgres.provider_admission import (
    PostgresProviderAdmission,
)
from trading.contexts.market_data.application.ports.provider_admission import (
    ProviderRequest,
    QuotaWindow,
)
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError

DSN = os.environ.get("ROEHUB_MARKET_DATA_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG required")


@pytest.fixture
def admission():
    schema = "provider_test_" + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        connection.execute(
            "CREATE TABLE market_data_history_bounds (market_id smallint, symbol text)"
        )
        connection.execute(
            sql.SQL(
                cast(
                    LiteralString,
                    Path(
                        "migrations/postgres/0034_market_data_provider_admission_v1.sql"
                    ).read_text(),
                )
            )
        )
    dsn = make_conninfo(DSN, options=f"-c search_path={schema}")
    try:
        yield PostgresProviderAdmission(dsn)
    finally:
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def request(*, concurrency=1, windows=None):
    return ProviderRequest(
        "binance",
        "disposable-egress",
        concurrency,
        windows or (QuotaWindow("spot_weight", 60, 10000, 2), QuotaWindow("raw", 300, 10000, 1)),
    )


def _hold(dsn, pipe):
    permit = PostgresProviderAdmission(dsn).reserve(request())
    pipe.send(str(permit.token))
    pipe.recv()
    permit.complete(headers={})
    pipe.send("stopped")


def test_two_processes_share_capacity_and_spent_quota_survives_restart(admission):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold, args=(admission.dsn, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        parent.recv()
        with pytest.raises(TemporarySourceError):
            admission.reserve(request())
        parent.send("stop")
        assert parent.poll(10) and parent.recv() == "stopped"
        process.join(10)
        assert process.exitcode == 0
        permit = PostgresProviderAdmission(admission.dsn).reserve(request())
        permit.complete(headers={})
        with psycopg.connect(admission.dsn) as connection:
            assert connection.execute(
                "SELECT spent FROM market_data_provider_windows " "WHERE dimension='spot_weight'"
            ).fetchone() == (4,)
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def test_multi_quota_reservation_is_atomic_and_headers_only_increase_spend(admission):
    policy = request(windows=(QuotaWindow("spot_weight", 60, 10, 2), QuotaWindow("raw", 300, 1, 1)))
    permit = admission.reserve(policy)
    permit.complete(headers={"X-MBX-USED-WEIGHT-1M": "8"})
    with pytest.raises(TemporarySourceError):
        admission.reserve(policy)
    with psycopg.connect(admission.dsn) as connection:
        assert dict(
            connection.execute(
                "SELECT dimension,spent FROM market_data_provider_windows"
            ).fetchall()
        ) == {"spot_weight": 8, "raw": 1}
        assert connection.execute(
            "SELECT count(*) FROM market_data_provider_permits"
        ).fetchone() == (0,)
    # Local clocks never enter reservations. A backwards DB clock fails closed.
    with psycopg.connect(admission.dsn) as connection:
        connection.execute("UPDATE market_data_provider_windows SET bucket=bucket+1")
    with pytest.raises(TemporarySourceError):
        admission.reserve(request())


@pytest.fixture
def provider():
    state = SimpleNamespace(
        started=threading.Event(),
        disconnected=threading.Event(),
        requests=0,
        connections=set(),
        statuses=[],
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            state.requests += 1
            state.connections.add(self.client_address)
            status = state.statuses.pop(0) if state.statuses else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 429:
                self.send_header("Retry-After", "2")
            trickle = self.path.startswith("/trickle")
            body = b"[]" if not trickle else b"[" + b" " * 10000 + b"]"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            state.started.set()
            try:
                if trickle:
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(body)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state.disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    thread.join()


def client(admission):
    return RequestsHttpClient(
        admission=admission, policy=SimpleNamespace(request=lambda *_: request())
    )


def call(http, url, timeout: float = 3, retries=0):
    return http.get_json(
        url=url,
        params={},
        timeout_s=timeout,
        retries=retries,
        backoff_base_s=0.01,
        backoff_max_s=0.01,
        backoff_jitter_s=0,
    )


def test_session_reuse_retries_charge_every_attempt_and_shared_retry_after(admission, provider):
    url, state = provider
    http = client(admission)
    try:
        state.statuses = [503, 200]
        assert call(http, url, retries=1).body == []
        assert call(http, url).body == []
        assert state.requests == 3 and len(state.connections) == 1
        state.statuses = [429]
        with pytest.raises(TemporarySourceError) as caught:
            call(http, url, retries=2)
        assert caught.value.retry_after_s == 2
        with pytest.raises(TemporarySourceError):
            call(client(admission), url)
        assert state.requests == 4
    finally:
        http.close()


def test_absolute_deadline_stops_trickle_before_permit_release(admission, provider):
    url, state = provider
    http = client(admission)
    started = time.monotonic()
    try:
        with pytest.raises(TemporarySourceError):
            call(http, url + "/trickle", timeout=1)
        assert time.monotonic() - started < 3
        assert all(t.closed for t in http._transports)
        assert state.disconnected.wait(2)
        permit = admission.reserve(request())
        permit.complete(headers={})
    finally:
        http.close()


def test_permit_session_loss_stops_io_before_capacity_is_reused(admission, provider):
    url, state = provider
    http = client(admission)
    original = admission.reserve
    permits = []

    def reserve(policy):
        permit = original(policy)
        permits.append(permit)
        return permit

    admission.reserve = reserve
    errors = []

    def run():
        try:
            call(http, url + "/trickle", timeout=5)
        except TemporarySourceError:
            errors.append("stopped")

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert state.started.wait(5)
        with psycopg.connect(admission.dsn, autocommit=True) as connection:
            connection.execute(
                "SELECT pg_terminate_backend(%s)", (permits[0].connection.info.backend_pid,)
            )
        thread.join(5)
        assert not thread.is_alive() and errors == ["stopped"]
        assert all(t.closed for t in http._transports)
        assert state.disconnected.wait(2)
        permit = original(request())
        permit.complete(headers={})
    finally:
        http.close()
        thread.join(5)


def test_coordination_outage_sends_no_http(admission, provider):
    url, state = provider
    unavailable = PostgresProviderAdmission(make_conninfo(admission.dsn, port=1))
    http = client(unavailable)
    try:
        with pytest.raises(TemporarySourceError):
            call(http, url)
        assert state.requests == 0
    finally:
        http.close()


def test_owner_death_keeps_uncertain_permit_until_confirmed_drain(admission):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold, args=(admission.dsn, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        token = parent.recv()
        process.kill()
        process.join(10)
        assert not process.is_alive()
        with pytest.raises(TemporarySourceError):
            PostgresProviderAdmission(admission.dsn).reserve(request())
        # Runbook recovery: all I/O for this exact disposable owner is now proven
        # stopped by OS join; deleting a live owner's permit would be unsafe.
        with psycopg.connect(admission.dsn) as connection:
            connection.execute("DELETE FROM market_data_provider_permits WHERE token=%s", (token,))
        permit = admission.reserve(request())
        permit.complete(headers={})
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def test_child_absolute_deadline_survives_blocked_parent_pg_check(admission, provider):
    url, state = provider
    http = client(admission)
    original = admission.reserve
    blocked, unblock, ended = threading.Event(), threading.Event(), threading.Event()

    def reserve(policy):
        permit = original(policy)
        original_check = permit.check

        def check():
            if state.started.is_set() and not blocked.is_set():
                blocked.set()
                assert unblock.wait(5)
            original_check()

        permit.check = check
        return permit

    admission.reserve = reserve

    def run():
        try:
            call(http, url + "/trickle", timeout=1)
        except TemporarySourceError:
            pass
        finally:
            ended.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert blocked.wait(3)
        assert state.disconnected.wait(2)  # Parent is still blocked; child owns deadline.
        assert not ended.is_set()
        with pytest.raises(TemporarySourceError):
            original(request())  # No TTL-based capacity reuse before confirmed join.
    finally:
        unblock.set()
        thread.join(5)
        http.close()
    assert ended.is_set()
    permit = original(request())
    permit.complete(headers={})


def _http_owner(dsn, url):
    http = client(PostgresProviderAdmission(dsn))
    try:
        call(http, url + "/trickle", timeout=30)
    finally:
        http.close()


def test_parent_death_stops_orphan_http_but_does_not_recycle_uncertain_permit(admission, provider):
    url, state = provider
    process = multiprocessing.get_context("spawn").Process(
        target=_http_owner, args=(admission.dsn, url)
    )
    process.start()
    try:
        assert state.started.wait(5)
        process.kill()
        process.join(5)
        assert not process.is_alive()
        assert state.disconnected.wait(3)
        with pytest.raises(TemporarySourceError):
            admission.reserve(request())
        with psycopg.connect(admission.dsn) as connection:
            # Runbook confirmed-drain cleanup is scoped to the exact dead owner.
            connection.execute(
                "DELETE FROM market_data_provider_permits WHERE owner_pid=%s", (process.pid,)
            )
        permit = admission.reserve(request())
        permit.complete(headers={})
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_transport_pool_reuses_children_across_distinct_live_threads(admission, provider):
    url, state = provider
    http = RequestsHttpClient(admission=admission,
                              policy=SimpleNamespace(request=lambda *_: request(concurrency=16)),
                              transport_capacity=2)
    release = threading.Event()
    threads, errors, pids = [], [], set()
    try:
        for _ in range(12):
            ready = threading.Event()

            def run(ready=ready):
                try:
                    assert call(http, url).body == []
                except Exception as error:
                    errors.append(type(error).__name__)
                finally:
                    ready.set()
                release.wait()

            thread = threading.Thread(target=run)
            threads.append(thread)
            thread.start()
            assert ready.wait(5)
            assert len(http._transports) == 1
            pids.add(http._transports[0].process.pid)
        assert not errors and len(pids) == len(state.connections) == 1
        children = [t.process for t in http._transports]
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
        http.close()
    assert all(child.poll() is not None for child in children)


def test_pool_wait_deadline_sends_no_extra_http_and_close_drains(admission, provider):
    url, state = provider
    http = RequestsHttpClient(admission=admission,
                              policy=SimpleNamespace(request=lambda *_: request(concurrency=3)),
                              transport_capacity=1)
    errors = []

    def run():
        try:
            call(http, url+'/trickle', timeout=10)
        except TemporarySourceError:
            errors.append('stopped')

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert state.started.wait(5)
        start = time.monotonic()
        with pytest.raises(TemporarySourceError):
            call(http, url, timeout=0.3)
        assert time.monotonic()-start < 1.5
        assert state.requests == len(http._transports) == 1
        children = [t.process for t in http._transports]
    finally:
        http.close()
        thread.join(5)
    assert errors == ['stopped'] and not thread.is_alive()
    assert state.disconnected.wait(2)
    assert all(child.poll() is not None for child in children)
    with psycopg.connect(admission.dsn) as connection:
        assert connection.execute(
            'SELECT count(*) FROM market_data_provider_permits'
        ).fetchone() == (0,)


def test_pool_close_wakes_waiter_without_new_child_or_http(admission, provider):
    url, state = provider
    http = RequestsHttpClient(admission=admission,
                              policy=SimpleNamespace(request=lambda *_: request(concurrency=3)),
                              transport_capacity=1)
    errors = []
    second_reserved = threading.Event()
    original = admission.reserve
    reservations = []

    def reserve(policy):
        permit = original(policy)
        reservations.append(permit)
        if len(reservations) == 2:
            second_reserved.set()
        return permit

    admission.reserve = reserve

    def run(path):
        try:
            call(http, url+path, timeout=10)
        except (TemporarySourceError, RuntimeError) as error:
            errors.append(type(error).__name__)

    first = threading.Thread(target=run, args=('/trickle',))
    second = threading.Thread(target=run, args=('',))
    first.start()
    try:
        assert state.started.wait(5)
        second.start()
        assert second_reserved.wait(5)
        http.close()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        assert sorted(errors) == ['RuntimeError', 'TemporarySourceError']
        assert state.requests == 1 and state.disconnected.wait(2)
        with psycopg.connect(admission.dsn) as connection:
            assert connection.execute(
                'SELECT count(*) FROM market_data_provider_permits'
            ).fetchone() == (0,)
    finally:
        http.close()
        first.join(5)
        if second.ident is not None:
            second.join(5)


def test_dead_idle_child_is_joined_and_replaced_without_capacity_leak(admission, provider):
    url, state = provider
    http = client(admission)
    try:
        call(http, url)
        child = http._transports[0].process
        child.kill()
        child.wait(5)
        with pytest.raises(TemporarySourceError):
            call(http, url)
        assert child.poll() is not None and not http._transports
        assert call(http, url).body == []
        assert http._transports[0].process.pid != child.pid
        assert state.requests == 2
    finally:
        http.close()


def test_managed_upgrade_applies_admission_once_and_preserves_spend(monkeypatch):
    """Exercise new SQL phase/marker; prior phases and Alembic are outside this fixture."""
    from apps.migrations import storage

    root = Path(__file__).resolve().parents[3]
    manifest = root / "migrations/postgres/manifest.json"
    phases = storage._load_postgres_phases(manifest)
    version = "market-data-provider-admission-0034"
    schema = "provider_upgrade_" + uuid4().hex
    dsn = make_conninfo(DSN, options=f"-c search_path={schema}")
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        with psycopg.connect(dsn) as connection:
            connection.execute(
                "CREATE TABLE market_data_history_bounds (market_id smallint, symbol text)"
            )
        storage._ensure_postgres_state_table(dsn)
        for prior, (checksum, _) in phases.items():
            if prior != version:
                storage._record_postgres_marker(
                    dsn, store="postgres-sql", version=prior, checksum=checksum
                )
        monkeypatch.setattr(storage, "_alembic_revision_files", lambda _: ())
        monkeypatch.setattr(storage, "run_alembic_upgrade_head", lambda **_: None)
        original_marker = storage._record_postgres_marker

        def crash_before_marker(*args, **kwargs):
            if kwargs.get("version") == version:
                raise RuntimeError("simulated death after DDL commit before marker")
            return original_marker(*args, **kwargs)

        monkeypatch.setattr(storage, "_record_postgres_marker", crash_before_marker)
        with pytest.raises(RuntimeError, match="simulated death"):
            storage.apply_postgres_migrations(dsn, repo_root=root, manifest_path=manifest)
        assert ("postgres-sql", version) not in storage._read_postgres_markers(dsn)
        permit = PostgresProviderAdmission(dsn).reserve(request())
        permit.complete(headers={})
        monkeypatch.setattr(storage, "_record_postgres_marker", original_marker)
        storage.apply_postgres_migrations(dsn, repo_root=root, manifest_path=manifest)
        storage.apply_postgres_migrations(dsn, repo_root=root, manifest_path=manifest)
        assert storage._read_postgres_markers(dsn)[("postgres-sql", version)] == phases[version][0]
        with psycopg.connect(dsn) as connection:
            assert connection.execute(
                "SELECT spent FROM market_data_provider_windows WHERE dimension='spot_weight'"
            ).fetchone() == (2,)
            assert connection.execute(
                "SELECT next_retry_at, error_code FROM market_data_history_bounds"
            ).fetchall() == []
            assert connection.execute(
                "SELECT count(*) FROM market_data_provider_permits"
            ).fetchone() == (0,)
    finally:
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
