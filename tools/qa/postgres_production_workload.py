"""Reproducible S09 mixed workload; never target the installed runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import resource
import threading
import time
import traceback
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request

from apps.api.routes.market_data_reference import build_market_data_reference_router
from tools.qa.postgres_production_soak import APP_DSN, DSN, OUT, candle, emit
from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.adapters.outbound.persistence.postgres.research_organization_scope import (  # noqa: E501
    PostgresResearchOrganizationScopeResolver,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    PsycopgIdentityPostgresGateway,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.local_auth_repository import (
    PostgresLocalAuthRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.session_repository import (
    PostgresIdentitySessionRepository,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.user_repository import (
    PostgresIdentityUserRepository,
)
from trading.contexts.identity.adapters.outbound.security.current_user.roehub_session_current_user import (  # noqa: E501
    RoehubSessionCurrentUser,
)
from trading.contexts.identity.adapters.outbound.time.system_identity_clock import (
    SystemIdentityClock,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
    AsyncPostgresStreamRecoveryStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_coverage_reader import (  # noqa: E501
    PostgresCatalogCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
    PostgresHistoryCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_readers import (
    PostgresEnabledMarketReader,
    PostgresEnabledTradableInstrumentSearchReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.services.work_requests import WorkRequest
from trading.contexts.market_data.application.use_cases import (
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.platform.postgres_budget import connect
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UserId,
    UtcTimestamp,
)

ACTOR = UUID("7baa23f5-4c71-4edf-a27f-d1519eec3ae9")
ORG = UUID("ff4c9f56-2f37-4b48-83a7-845cce82d44a")
INSTALL = UUID("3c957c47-a5f8-4f87-8d7e-f4e51e8f025c")
BASE = datetime(2030, 1, 1, tzinfo=UTC)
CUTOFF = datetime(2026, 10, 12, tzinfo=UTC)


def instrument(symbol, market=1):
    return InstrumentId(MarketId(market), Symbol(symbol))


def span(start, end):
    return TimeRange(UtcTimestamp(start), UtcTimestamp(end))


def replay(symbol, offset, count):
    return [
        candle(
            dict(
                market_id=1,
                symbol=symbol,
                instrument_key=f"binance:spot:{symbol}",
                ts_open=BASE + timedelta(minutes=i),
                ts_close=BASE + timedelta(minutes=i + 1),
                open=10.0 + (i % 100) * 0.01,
                high=12.0,
                low=9.0,
                close=11.0 + (i % 100) * 0.001,
                volume_base=2.0 + (i % 10),
                volume_quote=20.0,
                trades_count=3,
                taker_buy_volume_base=1.0,
                taker_buy_volume_quote=10.0,
                source="rest",
                ingested_at=BASE,
                ingest_id="20d9c9c5-b565-40e0-a03a-6cbd6d28df79",
            )
        )
        for i in range(offset, offset + count)
    ]


def array_hash(batch):
    h = hashlib.sha256()
    for a in (batch.open_time_ms, batch.close_time_ms, batch.ohlcv_f32):
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        for offset in range(0, len(a), 8192):
            h.update(a[offset : offset + 8192].tobytes(order="C"))
    return h.hexdigest()


def canonical_record(item):
    c, m = item.candle, item.meta
    return dict(
        market_id=c.instrument_id.market_id.value,
        symbol=str(c.instrument_id.symbol),
        instrument_key=m.instrument_key,
        ts_open=c.ts_open.value,
        ts_close=c.ts_close.value,
        open=c.open,
        high=c.high,
        low=c.low,
        close=c.close,
        volume_base=c.volume_base,
        volume_quote=c.volume_quote,
        trades_count=m.trades_count,
        taker_buy_volume_base=m.taker_buy_volume_base,
        taker_buy_volume_quote=m.taker_buy_volume_quote,
        source=m.source,
        ingested_at=m.ingested_at.value,
        ingest_id=m.ingest_id,
    )


def canonical_hash(value):
    def normalized(v):
        if isinstance(v, dict):
            return [
                [normalized(k), normalized(x)]
                for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))
            ]
        if isinstance(v, (tuple, list)):
            return [normalized(x) for x in v]
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, UUID):
            return str(v)
        return v

    return hashlib.sha256(
        json.dumps(normalized(value), separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seed():
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import (  # noqa: E501
        PostgresMarketRefWriter,
    )
    from trading.contexts.market_data.application.dto.reference_data import RefMarketRow

    now = datetime.now(UTC)
    rg = IngestionGateway(APP_DSN, workload="reference")
    try:
        PostgresMarketRefWriter(rg).insert(
            [
                RefMarketRow(MarketId(m), e, t, code, 1, 0, UtcTimestamp(now))
                for m, e, t, code in [
                    (1, "binance", "spot", "binance:spot"),
                    (2, "binance", "futures", "binance:futures"),
                    (3, "bybit", "spot", "bybit:spot"),
                    (4, "bybit", "futures", "bybit:futures"),
                ]
            ]
        )
    finally:
        rg.close()
    g = PsycopgIdentityPostgresGateway(dsn=APP_DSN)
    users = PostgresIdentityUserRepository(gateway=g)
    if users.find_by_user_id(user_id=UserId(ACTOR)) is None:
        users.create_local_user(user_id=UserId(ACTOR), created_at=now)
    with connect(DSN, budget_class="admin") as c:
        c.execute(
            (
                "INSERT INTO identity_installations(installation_id,display_name,create"
                "d_at) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING"
            ),
            (INSTALL, "S09 isolated", now),
        )
        c.execute(
            (
                "INSERT INTO identity_organizations(organization_id,installation_id,slu"
                "g,display_name,created_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTH"
                "ING"
            ),
            (ORG, INSTALL, "s09-proof", "S09 proof", now),
        )
        c.execute(
            (
                "INSERT INTO identity_memberships(organization_id,user_id,role,created_"
                "at,updated_at) VALUES (%s,%s,'owner',%s,%s) ON CONFLICT DO NOTHING"
            ),
            (ORG, ACTOR, now, now),
        )


def api():
    g = PsycopgIdentityPostgresGateway(dsn=APP_DSN)
    sessions = PostgresIdentitySessionRepository(gateway=g)
    users = PostgresIdentityUserRepository(gateway=g)
    session = sessions.create_session(
        user_id=UserId(ACTOR),
        now=datetime.now(UTC),
        idle_ttl_seconds=7200,
        absolute_ttl_seconds=7200,
    )
    resolver = RoehubSessionCurrentUser(
        session_repository=sessions, user_repository=users, clock=SystemIdentityClock()
    )

    def current(request: Request):
        return resolver.require(session_id=request.cookies.get("s09_session"))

    gateway = IngestionGateway(APP_DSN, workload="reference")
    app = FastAPI()
    app.include_router(
        build_market_data_reference_router(
            list_enabled_markets_use_case=ListEnabledMarketsUseCase(
                PostgresEnabledMarketReader(gateway)
            ),
            search_enabled_tradable_instruments_use_case=SearchEnabledTradableInstrumentsUseCase(
                PostgresEnabledTradableInstrumentSearchReader(gateway)
            ),
            current_user_dependency=current,
            organization_scope_resolver=PostgresResearchOrganizationScopeResolver(
                gateway=PsycopgBacktestPostgresGateway(dsn=APP_DSN)
            ),
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=18709, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    return server, thread, gateway, str(session.session_id)


class MeteredGateway(IngestionGateway):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local = threading.local()
        original = self._pool.get

        def get(*a, **kw):
            start = time.monotonic()
            try:
                return original(*a, **kw)
            finally:
                self.local.pool_wait_s = (
                    getattr(self.local, "pool_wait_s", 0) + time.monotonic() - start
                )

        self._pool.get = get


def run(name, mode, duration):
    assert duration > 0 and mode in ("normal", "stress")
    seed()
    target = json.loads((OUT / "read-target.json").read_text())
    iid = instrument(target["symbol"], target["market"])
    full = span(datetime.fromisoformat(target["first"]), CUTOFF)
    year = span(CUTOFF - timedelta(days=365), CUTOFF)
    expected = target["array_hashes"]
    # Every run uses a new deterministic symbol namespace; no retained receipt is deleted.
    suffix = hashlib.sha256(name.encode()).hexdigest()[:8].upper()
    stream_symbol = "S09S" + suffix
    history_symbol = "S09H" + suffix
    gateways = {
        k: MeteredGateway(APP_DSN, workload=w, deadline_s=30 if w == "bulk" else 5)
        for k, w in [
            ("stream", "ingestion"),
            ("history", "ingestion"),
            ("latest", "interactive"),
            ("hot", "interactive"),
            ("catalog", "interactive"),
            ("gap", "interactive"),
            ("year", "bulk"),
            ("full", "bulk"),
        ]
    }
    records = []
    lock = threading.Lock()
    start = time.monotonic() + 2
    server, api_thread, api_gateway, session_id = api()
    # HTTP session id exists only in process memory; never record credentials/cookies.
    client = httpx.Client(
        base_url="http://127.0.0.1:18709", cookies={"s09_session": session_id}, timeout=5
    )
    log = (OUT / f"{name}-samples.jsonl").open("x")

    def record(row):
        with lock:
            records.append(row)
            log.write(json.dumps(row, default=str) + "\n")
            log.flush()

    def lane(kind, interval, operation, gw=None):
        count = math.ceil(duration / interval)
        for i in range(count):
            due = start + i * interval
            time.sleep(max(0, due - time.monotonic()))
            begun = time.monotonic()
            cpu = time.thread_time()
            if gw:
                gw.local.pool_wait_s = 0
            row = {
                "lane": kind,
                "index": i,
                "due_s": i * interval,
                "start_s": begun - start,
                "queue_wait_s": max(0, begun - due),
            }
            try:
                result = operation(i)
                row.update(ok=True, result=result)
            except Exception as e:
                row.update(ok=False, error=type(e).__name__, sqlstate=getattr(e, "sqlstate", None))
                traceback.print_exc()
            row.update(
                elapsed_s=time.monotonic() - begun,
                end_s=time.monotonic() - start,
                client_thread_cpu_s=time.thread_time() - cpu,
                pool_wait_s=getattr(gw.local, "pool_wait_s", 0) if gw else None,
            )
            record(row)

    def write(kind, symbol, amount, i):
        batch = replay(symbol, i * amount, amount)
        # Exact input hashes are regenerated with receipt parity after the timed horizon.
        adapter_start = time.monotonic()
        PostgresRawKlineWriter(gateways[kind]).write_1m(batch)
        adapter_s = time.monotonic() - adapter_start
        return {
            "successful_rows": amount,
            "input_fixture": "s09-replay-v1",
            "offset": i * amount,
            "adapter_s": adapter_s,
        }

    def arrays(kind):
        adapter_start = time.monotonic()
        b = PostgresCanonicalCandleReader(gateways[kind]).read_1m_arrays(
            iid, year if kind == "year" else full
        )
        adapter_s = time.monotonic() - adapter_start
        # End-to-end array input includes checking every byte, separately reported below.
        t = time.monotonic()
        h = array_hash(b)
        check = time.monotonic() - t
        assert h == expected[kind], (kind, h, expected[kind])
        return {"rows": b.row_count(), "sha256": h, "hash_seconds": check, "adapter_s": adapter_s}

    # Seed one known active row before the timing horizon; its raw observation is
    # accounted separately.
    PostgresRawKlineWriter(gateways["stream"]).write_1m(replay(stream_symbol, 0, 1))

    def checked_hot(i):
        result = PostgresCanonicalCandleReader(gateways["hot"]).latest(
            instrument_id=instrument(stream_symbol),
            before=UtcTimestamp(BASE + timedelta(days=365)),
            limit=100,
        )
        assert result
        for row in result:
            offset = int((row["ts_open"] - BASE).total_seconds() / 60)
            expected_row = replay(stream_symbol, offset, 1)[0]
            c, m = expected_row.candle, expected_row.meta
            expected = dict(
                market_id=1,
                symbol=stream_symbol,
                instrument_key=m.instrument_key,
                ts_open=c.ts_open.value,
                ts_close=c.ts_close.value,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume_base=c.volume_base,
                volume_quote=c.volume_quote,
                trades_count=m.trades_count,
                taker_buy_volume_base=m.taker_buy_volume_base,
                taker_buy_volume_quote=m.taker_buy_volume_quote,
                source=m.source,
                ingested_at=m.ingested_at.value,
                ingest_id=m.ingest_id,
            )
            assert canonical_hash(row) == canonical_hash(expected)
        return {"rows": len(result), "sha256": canonical_hash(result)}

    def checked_latest(i):
        result = PostgresCanonicalCandleReader(gateways["latest"]).latest(
            instrument_id=iid, before=UtcTimestamp(CUTOFF), limit=100
        )
        digest = canonical_hash(result)
        assert digest == target["latest_sha256"]
        return {"rows": len(result), "sha256": digest}

    def checked_catalog(i):
        result = PostgresCatalogCoverageReader(gateways["catalog"]).counts(
            market_ids=[1, 2, 3, 4], start_at=CUTOFF - timedelta(days=7), end_at=CUTOFF
        )
        fixed = {k: v for k, v in result.items() if not k[1].startswith("S09")}
        assert canonical_hash(fixed) == target["catalog_sha256"]
        assert all(v == 0 for k, v in result.items() if k[1].startswith("S09"))
        return {"keys": len(result), "sha256": canonical_hash(result)}

    def checked_gap(i):
        result = PostgresHistoryCoverageReader(gateways["gap"]).first_gap(
            instrument_id=iid, time_range=span(CUTOFF - timedelta(days=7), CUTOFF)
        )
        value = (
            None
            if result is None
            else [result.start.value.isoformat(), result.end.value.isoformat()]
        )
        assert value == target["gap_expected"]
        return {"sha256": canonical_hash(value)}

    def http(i):
        r = client.get("/market-data/markets")
        r.raise_for_status()
        payload = r.json()
        expected_markets = {
            "items": [
                dict(market_id=m, exchange_name=e, market_type=t, market_code=code)
                for m, e, t, code in [
                    (1, "binance", "spot", "binance:spot"),
                    (2, "binance", "futures", "binance:futures"),
                    (3, "bybit", "spot", "bybit:spot"),
                    (4, "bybit", "futures", "bybit:futures"),
                ]
            ]
        }
        assert payload == expected_markets
        return {"status": r.status_code, "sha256": hashlib.sha256(r.content).hexdigest()}

    auth = PostgresLocalAuthRepository(dsn=APP_DSN)

    def identity(i):
        subject = hashlib.sha256(f"s09-{i%8}".encode()).hexdigest()
        now = datetime.now(UTC)
        auth.record_auth_failure(subject_sha256=subject, now=now)
        limited = auth.is_rate_limited(subject_sha256=subject, now=now)
        auth.clear_auth_failures(subject_sha256=subject)
        return {"rate_limited": limited, "write_read_clear": True}

    jobs = PostgresMarketDataWorkRequestRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=APP_DSN), execution_dsn=APP_DSN
    )

    def job(i):
        assert ownership is not None
        now = datetime.now(UTC)
        ownership()
        r = jobs.submit(
            organization_id=ORG,
            actor_user_id=ACTOR,
            key=uuid4(),
            request=WorkRequest(kind="catalog_refresh", market_id=1),
            now=now,
        )
        ownership()
        claimed = jobs.claim(now=now)
        assert claimed and claimed["job_id"] == r["job_id"]
        ownership()
        assert (
            jobs.checkpoint(
                job=claimed, units=1, rows_read=4, rows_written=4, now=datetime.now(UTC)
            )
            == "running"
        )
        ownership()
        jobs.finish(job=claimed, state="succeeded", error_code=None, now=datetime.now(UTC))
        final = jobs.get(organization_id=ORG, job_id=r["job_id"])
        assert final and final["state"] == "succeeded"
        return {"state": final["state"], "job_id": str(r["job_id"])}

    async def recovery():
        store = AsyncPostgresStreamRecoveryStore(APP_DSN)
        async with store.lease():
            for i in range(math.ceil(duration / 2)):
                due = start + i * 2
                await asyncio.sleep(max(0, due - time.monotonic()))
                begun = time.monotonic()
                row = {
                    "lane": "recovery",
                    "index": i,
                    "due_s": i * 2,
                    "start_s": begun - start,
                    "queue_wait_s": max(0, begun - due),
                }
                try:
                    task = RestFillTask(
                        iid, span(CUTOFF - timedelta(minutes=2), CUTOFF), "reconnect_tail"
                    )
                    entries = await store.remember_many([task])
                    assert len(entries) == 1
                    assert await store.pending()
                    await store.complete_many(entries)
                    await store.publish_connections({(target["market"], target["symbol"]): True})
                    row.update(
                        ok=True,
                        result={"receipts": 1, "pending_after_ack": len(await store.pending())},
                    )
                except Exception as e:
                    row.update(ok=False, error=type(e).__name__)
                    traceback.print_exc()
                row["elapsed_s"] = time.monotonic() - begun
                row["end_s"] = time.monotonic() - start
                record(row)

    calls = [
        ("hot", 1, checked_hot, gateways["hot"]),
        (
            "stream",
            1,
            lambda i: write("stream", stream_symbol, 5 if mode == "normal" else 100, i),
            gateways["stream"],
        ),
        ("history", 2, lambda i: write("history", history_symbol, 1000, i), gateways["history"]),
        (
            "latest",
            1 if mode == "normal" else 0.5,
            checked_latest,
            gateways["latest"],
        ),
        (
            "catalog",
            2,
            checked_catalog,
            gateways["catalog"],
        ),
        (
            "gap",
            2,
            checked_gap,
            gateways["gap"],
        ),
        ("year", 10, lambda i: arrays("year"), gateways["year"]),
        ("full", 10, lambda i: arrays("full"), gateways["full"]),
        ("http", 1, http, None),
        ("identity", 2, identity, None),
        ("job", 5, job, None),
    ]
    lifecycle_errors = []

    def guarded(kind, callback):
        try:
            callback()
        except BaseException as exc:
            lifecycle_errors.append({"lane": kind, "error": type(exc).__name__})
            traceback.print_exc()

    threads = [
        threading.Thread(target=guarded, args=(call[0], lambda call=call: lane(*call)))
        for call in calls
    ]
    threads.append(
        threading.Thread(target=guarded, args=("recovery", lambda: asyncio.run(recovery())))
    )
    start = time.monotonic() + 2
    window_start_utc = time.time() + start - time.monotonic()
    emit(
        name + "-window.json",
        {
            "start_utc": window_start_utc,
            "end_utc": window_start_utc + duration,
            "duration_s": duration,
        },
    )
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        with jobs.execution() as ownership:
            assert ownership is not None
            for thread in threads:
                thread.start()
            while any(thread.is_alive() for thread in threads):
                time.sleep(1)
                emit(
                    name + "-progress.json",
                    {
                        "elapsed_s": time.monotonic() - start,
                        "operations": len(records),
                        "errors": sum(not r["ok"] for r in records),
                        "client_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    },
                )
            for thread in threads:
                thread.join()
        time.sleep(max(0, start + duration - time.monotonic()))
    finally:
        for g in gateways.values():
            g.close()
        client.close()
        server.should_exit = True
        api_thread.join(10)
        api_gateway.close()
        log.close()
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "name": name,
        "mode": mode,
        "duration_requested_s": duration,
        "window_start_utc": window_start_utc,
        "window_end_utc": window_start_utc + duration,
        "wall_s": time.monotonic() - start,
        "client_cpu_s": usage1.ru_utime + usage1.ru_stime - usage0.ru_utime - usage0.ru_stime,
        "client_maxrss_kib": usage1.ru_maxrss,
        "stream_symbol": stream_symbol,
        "history_symbol": history_symbol,
        "lanes": {},
        "lifecycle_errors": lifecycle_errors,
    }
    for kind in sorted({c[0] for c in calls} | {"recovery"}):
        rs = [r for r in records if r["lane"] == kind]
        values = np.asarray(
            [
                r.get("result", {}).get("adapter_s", r["elapsed_s"])
                if isinstance(r.get("result"), dict)
                else r["elapsed_s"]
                for r in rs
            ]
        )
        if not rs:
            summary["lanes"][kind] = {
                "offered": math.ceil(duration / next((c[1] for c in calls if c[0] == kind), 2)),
                "completed": 0,
                "missing": True,
            }
            continue
        summary["lanes"][kind] = {
            "offered": math.ceil(duration / next((c[1] for c in calls if c[0] == kind), 2)),
            "completed": sum(r["ok"] for r in rs),
            "completed_in_window": sum(r["ok"] and r["end_s"] <= duration for r in rs),
            "max_end_lag_s": max(r["end_s"] - r["due_s"] for r in rs),
            "last_queue_wait_s": rs[-1]["queue_wait_s"],
            "errors": sum(not r["ok"] for r in rs),
            "median_s": float(np.median(values)),
            "p95_s": float(np.quantile(values, 0.95)),
            "max_s": float(values.max()),
            "max_queue_wait_s": max(r["queue_wait_s"] for r in rs),
        }
    emit(name + "-summary.json", summary)
    print(json.dumps(summary), flush=True)
    assert not lifecycle_errors
    assert all(
        not r.get("missing") and r["completed"] == r["offered"] and not r["errors"]
        for r in summary["lanes"].values()
    ), "incomplete workload"


def verify_replay(name):
    from psycopg.rows import dict_row

    summary = json.loads((OUT / (name + "-summary.json")).read_text())
    proof = {}
    with connect(DSN, budget_class="admin") as c:
        for lane, symbol, amount in [
            ("stream", summary["stream_symbol"], 5 if summary["mode"] == "normal" else 100),
            ("history", summary["history_symbol"], 1000),
        ]:
            expected_count = summary["lanes"][lane]["completed"] * amount
            h_in = hashlib.sha256()
            h_out = hashlib.sha256()
            offset = 0
            receipts = []
            with c.cursor(name="replay_check", row_factory=dict_row, binary=True) as cursor:
                cursor.execute(
                    "SELECT market_id,symbol,instrument_key,ts_open,ts_close,open,high,low,close,"
                    "volume_base,volume_quote,trades_count,taker_buy_volume_base,"
                    "taker_buy_volume_quote,source,ingested_at,ingest_id "
                    "FROM market_data.canonical_candles_1m "
                    "WHERE market_id=1 AND symbol=%s ORDER BY ts_open",
                    (symbol,),
                )
                while rows := cursor.fetchmany(amount):
                    expected = [canonical_record(x) for x in replay(symbol, offset, len(rows))]
                    assert rows == expected, (name, lane, offset)
                    input_hash = canonical_hash(expected)
                    output_hash = canonical_hash(rows)
                    receipts.append(
                        {
                            "offset": offset,
                            "rows": len(rows),
                            "input_value_hash": input_hash,
                            "output_value_hash": output_hash,
                        }
                    )
                    h_in.update(input_hash.encode())
                    h_out.update(output_hash.encode())
                    offset += len(rows)
            assert offset == expected_count, (lane, offset, expected_count)
            raw = c.execute(
                "SELECT count(*) FROM market_data.raw_binance_klines_1m "
                "WHERE market_id=1 AND symbol=%s",
                (symbol,),
            ).fetchone()
            assert raw is not None and raw[0] == expected_count + (1 if lane == "stream" else 0), (
                lane,
                raw,
                expected_count,
            )
            proof[lane] = {
                "canonical_rows": offset,
                "raw_rows": raw[0],
                "input_sha256": h_in.hexdigest(),
                "output_sha256": h_out.hexdigest(),
                "all_17_fields_equal": True,
                "per_batch_receipts": receipts,
            }
    emit(name + "-replay-parity.json", proof)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--mode", choices=["normal", "stress"], default="normal")
    p.add_argument("--duration", type=int, default=30)
    a = p.parse_args()
    run(a.name, a.mode, a.duration)
    emit(a.name + "-verification-window.json", {"start_utc": time.time()})
    verify_replay(a.name)
    verification = json.loads((OUT / (a.name + "-verification-window.json")).read_text())
    verification["end_utc"] = time.time()
    emit(a.name + "-verification-window.json", verification)
