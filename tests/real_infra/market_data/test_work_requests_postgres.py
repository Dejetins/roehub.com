"""Opt-in real PostgreSQL proof in an isolated, disposable schema only."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, LiteralString, Mapping, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_bounds_repository import (  # noqa: E501
    PostgresHistoryBoundsRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.services.work_requests import WorkRequest

DSN = os.environ.get("ROEHUB_MARKET_DATA_TEST_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL target required")


@pytest.fixture
def store():
    schema = "work_requests_test_" + uuid4().hex
    with psycopg.connect(DSN) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        conn.execute(
            "CREATE TABLE identity_organizations "
            "(organization_id uuid PRIMARY KEY, status text DEFAULT $$active$$)"
        )
        conn.execute(
            "CREATE TABLE identity_memberships "
            "(organization_id uuid, user_id uuid, status text DEFAULT $$active$$, "
            "PRIMARY KEY (organization_id, user_id))"
        )
        for filename in (
            "0023_market_data_work_requests_v1.sql",
            "0024_market_data_full_history_v1.sql",
            "0025_market_data_work_recovery_v1.sql",
            "0026_market_data_queue_events_v1.sql",
        ):
            migration = (
                Path(__file__).resolve().parents[3] / "migrations/postgres" / filename
            ).read_text()
            conn.execute(
                sql.SQL(cast(LiteralString, migration.replace("BEGIN;", "").replace("COMMIT;", "")))
            )
        org, actor = uuid4(), uuid4()
        conn.execute("INSERT INTO identity_organizations (organization_id) VALUES (%s)", (org,))
        conn.execute(
            "INSERT INTO identity_memberships (organization_id,user_id) VALUES (%s,%s)",
            (org, actor),
        )

    class Gateway:
        def connect(self):
            conn = psycopg.Connection[dict[str, Any]].connect(DSN, row_factory=dict_row)
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            return conn

        def fetch_one(
            self, *, query: str, parameters: Mapping[str, Any]
        ) -> Mapping[str, Any] | None:
            with self.connect() as conn:
                return conn.execute(sql.SQL(cast(LiteralString, query)), parameters).fetchone()

        def fetch_all(
            self, *, query: str, parameters: Mapping[str, Any]
        ) -> tuple[Mapping[str, Any], ...]:
            with self.connect() as conn:
                return tuple(
                    conn.execute(sql.SQL(cast(LiteralString, query)), parameters).fetchall()
                )

        def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
            with self.connect() as conn:
                conn.execute(sql.SQL(cast(LiteralString, query)), parameters)

    try:
        gateway = Gateway()
        yield (
            PostgresMarketDataWorkRequestRepository(
                gateway=gateway, execution_dsn=DSN, execution_lock_key=uuid4().int % (2**63)
            ),
            org,
            actor,
            gateway,
        )
    finally:
        with psycopg.connect(DSN) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_atomic_duplicate_claim_cancel_retry_and_scope(store):
    repo, org, actor, _gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    command = WorkRequest("candle_ingestion", 2, ("BTCUSDT",), now - timedelta(minutes=5), now)
    key = uuid4()

    def submit(_):
        return repo.submit(
            organization_id=org, actor_user_id=actor, key=key, request=command, now=now
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(submit, range(4)))
    assert len({r["job_id"] for r in rows}) == 1
    job_id = rows[0]["job_id"]
    assert repo.backlog(organization_id=org)["queued"] == 1
    assert repo.backlog(organization_id=uuid4())["queued"] == 0
    assert repo.get(organization_id=uuid4(), job_id=job_id) is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: repo.claim(now=now), range(4)))
    assert sum(r is not None for r in claimed) == 1
    job = next(r for r in claimed if r is not None)
    assert repo.cancel(organization_id=org, job_id=job_id, now=now)["state"] == "cancel_requested"
    assert (
        repo.checkpoint(job=job, units=2, rows_read=2, rows_written=2, now=now)
        == "cancel_requested"
    )
    repo.finish(job=job, state="succeeded", error_code=None, now=now)
    assert repo.get(organization_id=org, job_id=job_id)["state"] == "cancelled"
    assert repo.retry(organization_id=org, job_id=job_id, attempt=1, now=now) is None
    next_row = repo.submit(
        organization_id=org, actor_user_id=actor, key=uuid4(), request=command, now=now
    )
    running = repo.claim(now=now)
    repo.finish(job=running, state="failed", error_code="source_or_storage_unavailable", now=now)
    assert (
        repo.retry(organization_id=org, job_id=next_row["job_id"], attempt=1, now=now)["attempt"]
        == 2
    )
    assert repo.retry(organization_id=org, job_id=next_row["job_id"], attempt=1, now=now) is None
    assert (
        repo.get(organization_id=org, job_id=next_row["job_id"])["attempts"][0]["state"] == "failed"
    )


def test_terminal_history_does_not_block_membership_revocation(store):
    repo, org, actor, gateway = store
    now = datetime.now(UTC)
    job = repo.submit(
        organization_id=org,
        actor_user_id=actor,
        key=uuid4(),
        request=WorkRequest("catalog_refresh", 2),
        now=now,
    )
    repo.cancel(organization_id=org, job_id=job["job_id"], now=now)
    gateway.execute(
        query=(
            "DELETE FROM identity_memberships "
            "WHERE organization_id=%(org)s AND user_id=%(actor)s"
        ),
        parameters=dict(org=org, actor=actor),
    )
    assert repo.get(organization_id=org, job_id=job["job_id"])["actor_user_id"] == actor
    with pytest.raises(PermissionError):
        repo.submit(
            organization_id=org,
            actor_user_id=actor,
            key=uuid4(),
            request=WorkRequest("catalog_refresh", 2),
            now=now,
        )


def test_live_slow_consumer_prevents_recovery_and_lost_session_fails_closed(store):
    repo, org, actor, gateway = store
    now = datetime.now(UTC)
    row = repo.submit(
        organization_id=org,
        actor_user_id=actor,
        key=uuid4(),
        request=WorkRequest("catalog_refresh", 2),
        now=now,
    )
    with repo.execution() as check:
        assert check is not None
        job = repo.claim(now=now)
        # Advance persisted heartbeat age past recovery threshold without sleeping.
        gateway.execute(
            query="UPDATE market_data_work_requests SET updated_at=%(at)s WHERE job_id=%(id)s",
            parameters=dict(at=now - timedelta(minutes=6), id=job["job_id"]),
        )
        with repo.execution() as second:
            assert second is None
        assert repo.get(organization_id=org, job_id=row["job_id"])["state"] == "running"
        check()
    # The old closure cannot authorize another write after its exact lock session closes.
    with pytest.raises(psycopg.Error):
        check()
    with repo.execution() as successor:
        assert successor is not None
        repo.recover(now=now)
        assert repo.claim(now=now) is None
        assert repo.get(organization_id=org, job_id=row["job_id"])["error_code"] == "worker_lost"
        resumed = repo.claim(now=now + timedelta(seconds=15))
        assert resumed["job_id"] == row["job_id"] and resumed["attempt"] == 1


def test_full_history_resume_and_cancel_at_yield(store):
    repo, org, actor, gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    row = repo.submit(
        organization_id=org,
        actor_user_id=actor,
        key=uuid4(),
        request=WorkRequest(
            "candle_ingestion", 2, ("BTCUSDT",), datetime(2017, 1, 1, tzinfo=UTC), now
        ),
        now=now,
    )
    assert row["total_units"] > 10080
    job = repo.claim(now=now)
    assert (
        repo.checkpoint(job=job, units=600, rows_read=600, rows_written=590, now=now) == "running"
    )
    repo.release(job=job, now=now)
    resumed = repo.claim(now=now + timedelta(seconds=2))
    assert resumed["completed_units"] == 600 and resumed["rows_written"] == 590
    assert resumed["started_at"] == now and resumed["attempt"] == 1
    # A cancellation between the last checkpoint and yielding must become terminal.
    repo.cancel(organization_id=org, job_id=row["job_id"], now=now)
    repo.release(job=resumed, now=now)
    done = repo.get(organization_id=org, job_id=row["job_id"])
    assert done["state"] == "cancelled" and done["attempts"][-1]["state"] == "cancelled"
    assert done["completed_units"] == 600 and done["worker_token"] is None
    with pytest.raises(psycopg.errors.CheckViolation):
        gateway.execute(
            query="UPDATE market_data_work_requests SET start_at=NULL WHERE job_id=%(id)s",
            parameters={"id": row["job_id"]},
        )


def test_history_probe_admission_is_bounded_under_concurrent_reads(store):
    _, _, _, gateway = store
    repo = PostgresHistoryBoundsRepository(gateway=gateway)
    now = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(
            pool.map(
                lambda i: repo.read_or_request(market_id=1, symbol=f"COIN{i}USDT", now=now),
                range(80),
            )
        )
    count = gateway.fetch_one(
        query="SELECT count(*) AS n FROM market_data_history_bounds", parameters={}
    )
    assert count is not None and count["n"] == 64
    job = repo.claim(now=now)
    assert job is not None
    repo.finish(job=job, first_open_at=datetime(2020, 1, 1, tzinfo=UTC), now=now)
    ready = repo.read_or_request(market_id=job["market_id"], symbol=job["symbol"], now=now)
    assert ready is not None and ready["state"] == "ready"
    admitted = repo.read_or_request(market_id=1, symbol="EXTRAUSDT", now=now)
    assert admitted is not None and admitted["state"] == "queued"


def test_active_full_history_retry_wins_over_a_newer_short_terminal_job(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    full = repo.submit(
        organization_id=org,
        actor_user_id=actor,
        key=uuid4(),
        request=WorkRequest(
            "candle_ingestion", 2, ("BTCUSDT",), datetime(2017, 1, 1, tzinfo=UTC), now
        ),
        now=now,
    )
    repo.finish(
        job=repo.claim(now=now), state="failed", error_code="source_or_storage_unavailable", now=now
    )
    later = now + timedelta(seconds=1)
    repo.submit(
        organization_id=org,
        actor_user_id=actor,
        key=uuid4(),
        request=WorkRequest("candle_ingestion", 2, ("BTCUSDT",), now - timedelta(minutes=5), now),
        now=later,
    )
    repo.finish(job=repo.claim(now=later), state="succeeded", error_code=None, now=later)
    repo.retry(
        organization_id=org, job_id=full["job_id"], attempt=1, now=later + timedelta(seconds=1)
    )
    visible = repo.instrument_jobs(
        organization_id=org, market_ids=[2], start_at=now - timedelta(minutes=5), end_at=now
    )
    assert (
        len(visible) == 1
        and visible[0]["job_id"] == full["job_id"]
        and visible[0]["state"] == "queued"
    )


def test_pause_resume_wait_cancel_are_durable_and_fenced(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    row = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                      request=WorkRequest('candle_ingestion', 2, ('BTCUSDT',),
                                          now - timedelta(days=90), now), now=now)
    job = repo.claim(now=now)
    repo.checkpoint(job=job, units=60, rows_read=60, rows_written=57, now=now)
    assert repo.pause(organization_id=uuid4(), job_id=row['job_id'], version=0, now=now) is None
    assert repo.pause(organization_id=org, job_id=row['job_id'], version=0,
                      now=now)['state'] == 'pause_requested'
    # Failure during the current provider request cannot override the user's pause.
    repo.defer(job=job, error_code='source_rate_limited', retry_after_s=90, now=now)
    paused = repo.get(organization_id=org, job_id=row['job_id'])
    assert paused['state'] == 'paused' and paused['completed_units'] == 60
    assert repo.claim(now=now + timedelta(days=1)) is None
    assert repo.resume(organization_id=org, job_id=row['job_id'], version=0, now=now) is None
    resumed = repo.resume(organization_id=org, job_id=row['job_id'], version=1, now=now)
    assert resumed['state'] == 'retry_wait' and resumed['rows_written'] == 57
    assert repo.claim(now=now + timedelta(seconds=89)) is None
    new = repo.claim(now=now + timedelta(seconds=90))
    assert new['job_id'] == job['job_id'] and new['attempt'] == 1
    # The abandoned worker token cannot overwrite the resumed checkpoint.
    assert repo.checkpoint(job=job, units=120, rows_read=120, rows_written=117,
                           now=now) is None
    repo.cancel(organization_id=org, job_id=row['job_id'], now=now)
    repo.defer(job=new, error_code='source_unavailable', retry_after_s=0, now=now)
    cancelled = repo.get(organization_id=org, job_id=row['job_id'])
    assert cancelled['state'] == 'cancelled' and cancelled['next_retry_at'] is None
    assert cancelled['completed_units'] == 60 and cancelled['rows_written'] == 57
    assert repo.claim(now=now + timedelta(days=1)) is None


def test_backoff_grows_without_resetting_progress_and_resets_after_progress(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC)
    row = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                      request=WorkRequest('catalog_refresh', 2), now=now)
    for delay in (15, 30, 60, 120, 240, 300, 300):
        job = repo.claim(now=now)
        repo.defer(job=job, error_code='source_unavailable', retry_after_s=0, now=now)
        waiting = repo.get(organization_id=org, job_id=row['job_id'])
        assert waiting['next_retry_at'] == now + timedelta(seconds=delay)
        assert waiting['attempt'] == 1 and waiting['attempts'] == []
        assert repo.claim(now=now + timedelta(seconds=delay-1)) is None
        now += timedelta(seconds=delay)
    job = repo.claim(now=now)
    repo.checkpoint(job=job, units=1, rows_read=0, rows_written=0, now=now)
    repo.defer(job=job, error_code='source_unavailable', retry_after_s=0, now=now)
    assert repo.get(organization_id=org, job_id=row['job_id'])['next_retry_at'] == (
        now + timedelta(seconds=15)
    )


def test_pause_and_resume_commands_do_not_replay_across_cycles(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC)
    row = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                      request=WorkRequest('catalog_refresh', 2), now=now)
    def pause(_):
        return repo.pause(organization_id=org, job_id=row['job_id'], version=0, now=now)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(pause, range(4)))
    assert sum(r is not None for r in results) == 1
    assert repo.get(organization_id=org, job_id=row['job_id'])['state'] == 'paused'
    repo.resume(organization_id=org, job_id=row['job_id'], version=1, now=now)
    repo.pause(organization_id=org, job_id=row['job_id'], version=2, now=now)
    assert repo.resume(organization_id=org, job_id=row['job_id'], version=1, now=now) is None
    assert repo.claim(now=now + timedelta(days=1)) is None


def test_queue_skips_pause_and_provider_wait_and_continues_without_reset(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    def submit(symbol):
        return repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                           request=WorkRequest('candle_ingestion', 1, (symbol,),
                                               now-timedelta(days=30), now), now=now)
    a = submit('AAAUSDT')
    repo.pause(organization_id=org, job_id=a['job_id'], version=0, now=now)
    b = submit('BBBUSDT')
    c = submit('CCCUSDT')
    job = repo.claim(now=now)
    repo.checkpoint(job=job, units=60, rows_read=60, rows_written=60, now=now)
    repo.defer(job=job, error_code='source_rate_limited', retry_after_s=90, now=now)
    other = repo.claim(now=now)
    assert other['job_id'] != job['job_id'] and other['job_id'] in {b['job_id'],c['job_id']}
    repo.checkpoint(job=other, units=120, rows_read=120, rows_written=120, now=now)
    repo.finish(job=other,state='failed',error_code='source_or_storage_unavailable',now=now)
    continued=repo.retry(organization_id=org,job_id=other['job_id'],attempt=1,now=now)
    counters = tuple(continued[k] for k in ['completed_units','rows_read','rows_written'])
    assert counters == (120,120,120)
    assert continued['started_at'] == other['started_at']
    assert repo.claim(now=now)['job_id'] == other['job_id']
    events=repo.events(organization_id=org,job_id=other['job_id'],before=None,limit=30)
    assert [e['event_type'] for e in events] == ['continued','failed','started','queued']
    assert repo.events(organization_id=uuid4(),job_id=other['job_id'],before=None,limit=30)==()


def test_concurrent_overlapping_admission_and_global_storage_cooldown(store):
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
        ActiveWorkRequestError,
    )
    repo,org,actor,_=store
    now=datetime.now(UTC).replace(second=0,microsecond=0)
    request=WorkRequest('candle_ingestion',1,('BTCUSDT',),now-timedelta(days=30),now)
    def submit(_):
        try:
            return repo.submit(organization_id=org,actor_user_id=actor,key=uuid4(),
                               request=request,now=now)
        except ActiveWorkRequestError as e:
            return e.code
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(submit,range(4)))
    assert results.count('duplicate_work')==3
    repo.submit(organization_id=org,actor_user_id=actor,key=uuid4(),
                request=WorkRequest('candle_ingestion',1,('ETHUSDT',),now-timedelta(days=30),now),now=now)
    job=repo.claim(now=now)
    repo.checkpoint(job=job,units=60,rows_read=60,rows_written=60,now=now)
    repo.defer(job={**job,'error_phase':'fill_window'},error_code='storage_memory_pressure',retry_after_s=0,now=now)
    assert repo.claim(now=now+timedelta(seconds=14)) is None
    assert repo.claim(now=now+timedelta(seconds=15)) is not None
    saved=repo.get(organization_id=org,job_id=job['job_id'])
    assert saved['completed_units']==60
    event=repo.events(organization_id=org,job_id=job['job_id'],before=None,limit=1)[0]
    assert event['event_type'] in {'retry_scheduled','recovered'}
    assert 'worker_token' not in event


def test_events_do_not_log_checkpoints_and_are_bounded_and_paginated(store):
    repo,org,actor,_=store
    now=datetime.now(UTC).replace(second=0,microsecond=0)
    row=repo.submit(organization_id=org,actor_user_id=actor,key=uuid4(),
                    request=WorkRequest('candle_ingestion',1,('BTCUSDT',),now-timedelta(days=30),now),now=now)
    job=repo.claim(now=now)
    for n in range(1,4):
        repo.checkpoint(job=job,units=n,rows_read=n,rows_written=n,now=now)
    repo.release(job=job,now=now)
    job=repo.claim(now=now)
    assert len(repo.events(organization_id=org,job_id=row['job_id'],before=None,limit=100))==2
    for n in range(260):
        repo.defer(job=job,error_code='source_unavailable',retry_after_s=0,now=now)
        now+=timedelta(seconds=301)
        job=repo.claim(now=now)
    first=repo.events(organization_id=org,job_id=row['job_id'],before=None,limit=300)
    second=repo.events(organization_id=org,job_id=row['job_id'],before=first[-1]['event_id'],limit=300)
    assert len(first)+len(second)==512
    assert not ({e['event_id'] for e in first}&{e['event_id'] for e in second})


def test_atomic_batches_idempotency_rollback_and_capacity(store):
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
        ActiveWorkRequestError,
    )
    repo, org, actor, gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)

    def commands(start, size=50):
        return tuple(WorkRequest('candle_ingestion', 1, (f'TEST{i}USDT',),
                                 now-timedelta(days=30), now) for i in range(start, start+size))

    key = uuid4()
    def submit(_):
        return repo.submit_batch(organization_id=org, actor_user_id=actor, key=key,
                                 requests=commands(0), now=now)
    with ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(submit, range(3)))
    assert len(groups[0]) == 50
    assert [{r['job_id'] for r in rows} for rows in groups].count(
        {r['job_id'] for r in groups[0]}) == 3
    assert repo.lookup(organization_id=org, key=key)['job_id'] == groups[0][0]['job_id']
    assert repo.lookup(organization_id=uuid4(), key=key) is None
    with pytest.raises(ValueError):
        repo.submit_batch(organization_id=org, actor_user_id=actor, key=key,
                          requests=commands(50), now=now)
    # New commands preceding a conflicting symbol must also roll back.
    with pytest.raises(ActiveWorkRequestError):
        repo.submit_batch(organization_id=org, actor_user_id=actor, key=uuid4(),
                          requests=commands(50, 1)+commands(0, 1), now=now)
    assert repo.backlog(organization_id=org)['queued'] == 50
    for offset in (50, 100, 150):
        repo.submit_batch(organization_id=org, actor_user_id=actor, key=uuid4(),
                          requests=commands(offset), now=now)
    with pytest.raises(ActiveWorkRequestError, match='queue_full'):
        repo.submit_batch(organization_id=org, actor_user_id=actor, key=uuid4(),
                          requests=commands(200), now=now)
    assert repo.backlog(organization_id=org)['queued'] == 200
    with pytest.raises(PermissionError):
        repo.submit_batch(organization_id=org, actor_user_id=uuid4(), key=uuid4(),
                          requests=commands(250), now=now)
    batches = gateway.fetch_one(query='SELECT count(*) n FROM market_data_work_batches',
                                parameters={})
    assert batches['n'] == 4


def test_pause_wins_terminal_error_and_active_job_ignores_catalog_period(store):
    repo, org, actor, _ = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    job = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                      request=WorkRequest('candle_ingestion', 1, ('BTCUSDT',),
                                          now-timedelta(days=60), now-timedelta(days=30)), now=now)
    running = repo.claim(now=now)
    repo.pause(organization_id=org, job_id=job['job_id'], version=0, now=now)
    repo.finish(job=running, state='failed', error_code='source_or_storage_unavailable', now=now)
    paused = repo.get(organization_id=org, job_id=job['job_id'])
    assert paused['state'] == 'paused' and paused['finished_at'] is None
    assert paused['attempts'] == []
    summary = repo.instrument_jobs(organization_id=org, market_ids=[1],
                                   start_at=now-timedelta(days=1), end_at=now)
    assert summary[0]['job_id'] == job['job_id']


@pytest.mark.skipif(not os.environ.get('ROEHUB_MARKET_DATA_TEST_CH_PORT'),
                    reason='explicit disposable ClickHouse target required')
def test_real_clickhouse_241_recovery_preserves_confirmed_checkpoint(store):
    from dataclasses import dataclass
    from types import SimpleNamespace

    import clickhouse_connect

    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.work_errors import (
        transient_clickhouse_error,
    )
    from trading.contexts.market_data.application.services.work_requests import (
        MarketDataWorkRequestRunner,
    )
    repo, org, actor, _ = store
    ch = clickhouse_connect.get_client(
        host='127.0.0.1', port=int(os.environ['ROEHUB_MARKET_DATA_TEST_CH_PORT']),
        username='proof', password=os.environ['ROEHUB_MARKET_DATA_TEST_CH_PASSWORD'])
    table = 'market_data.work_recovery_test_'+uuid4().hex
    clock = [datetime.now(UTC).replace(second=0, microsecond=0)]
    start = clock[0]-timedelta(hours=3)
    job = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                      request=WorkRequest('candle_ingestion', 1, ('BTCUSDT',), start, clock[0]),
                      now=clock[0])
    visits = []
    fault = [True]

    class Writer:
        def write_1m(self, rows):
            offset = rows[0][0]
            if offset == 60 and fault[0]:
                fault[0] = False
                ch.command(f'INSERT INTO {table} SELECT number FROM numbers(100000)',
                           settings={'max_memory_usage':1,'max_threads':1,'max_execution_time':5})
                pytest.fail('ClickHouse did not enforce the per-query memory limit')
            ch.insert(table, rows, column_names=['minute'])

    @dataclass
    class Fill:
        writer: Any

        def run(self, task):
            offset = int((task.time_range.start.value-start).total_seconds()//60)
            visits.append(offset)
            self.writer.write_1m([[i] for i in range(offset, offset+60)])
            return SimpleNamespace(rows_read=60, rows_written=60)

    try:
        ch.command(f'CREATE TABLE {table} (minute UInt64) ENGINE=MergeTree ORDER BY minute')
        runner = MarketDataWorkRequestRunner(
            repo, cast(Any, Fill(Writer())), lambda *_: None, now=lambda: clock[0],
            retryable_storage_error=lambda e: transient_clickhouse_error(e) or False,
        )
        assert runner.run_once()
        waiting = repo.get(organization_id=org, job_id=job['job_id'])
        assert (waiting['state'], waiting['completed_units'], waiting['rows_written']) == (
            'retry_wait', 60, 60)
        assert waiting['error_code'] == 'storage_memory_pressure'
        assert waiting['error_phase'] == 'fill_window'
        assert not runner.run_once()
        clock[0] = waiting['next_retry_at']
        assert runner.run_once()
        done = repo.get(organization_id=org, job_id=job['job_id'])
        assert (done['state'], done['completed_units'], done['rows_written']) == (
            'succeeded', 180, 180)
        assert visits == [0, 60, 60, 120]
        assert ch.query(f'SELECT count(),uniqExact(minute) FROM {table}').result_rows == [(180,180)]
        events = repo.events(organization_id=org, job_id=job['job_id'], before=None, limit=20)
        assert [e['event_type'] for e in events] == [
            'succeeded', 'recovered', 'retry_scheduled', 'started', 'queued']
    finally:
        ch.command(f'DROP TABLE IF EXISTS {table}')
        ch.close()


def test_parallel_batch_has_four_slots_and_preserves_pause_cancel_and_live_siblings(store):
    from dataclasses import dataclass
    from threading import Barrier, Event
    from types import SimpleNamespace

    from trading.contexts.market_data.application.services.work_requests import (
        MarketDataWorkRequestRunner,
    )

    repo, org, actor, gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    jobs = [repo.submit(
        organization_id=org, actor_user_id=actor, key=uuid4(),
        request=WorkRequest('candle_ingestion', 1, (symbol,), now-timedelta(hours=2), now),
        now=now+timedelta(microseconds=i),
    ) for i, symbol in enumerate(('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT'))]
    entered, release = Barrier(5), Event()

    @dataclass
    class Fill:
        writer: Any = None

        def run(self, task):
            entered.wait(timeout=10)
            assert release.wait(timeout=10)
            return SimpleNamespace(rows_read=60, rows_written=60)

    runner = MarketDataWorkRequestRunner(repo, cast(Any, Fill()), lambda *_: None,
                                        now=lambda: now, max_windows=1)
    with ThreadPoolExecutor(max_workers=4) as workers, ThreadPoolExecutor(1) as coordinator:
        future = coordinator.submit(runner.run_batch, executor=workers, concurrency=4)
        try:
            entered.wait(timeout=10)
            assert [repo.get(organization_id=org, job_id=j['job_id'])['state']
                    for j in jobs] == ['running']*4+['queued']
            with repo.execution() as rival:
                assert rival is None
            # An old sibling heartbeat must not be recovered by another claim.
            gateway.execute(
                query='UPDATE market_data_work_requests SET updated_at=%(at)s WHERE job_id=%(id)s',
                parameters=dict(at=now-timedelta(minutes=6), id=jobs[2]['job_id']),
            )
            repo.cancel(organization_id=org, job_id=jobs[0]['job_id'], now=now)
            paused = repo.get(organization_id=org, job_id=jobs[1]['job_id'])
            repo.pause(organization_id=org, job_id=jobs[1]['job_id'],
                       version=paused['control_version'], now=now)
        finally:
            release.set()
        assert future.result(timeout=10) == 4
    rows = [repo.get(organization_id=org, job_id=j['job_id']) for j in jobs]
    assert [r['state'] for r in rows] == ['cancelled', 'paused', 'queued', 'queued', 'queued']
    assert [r['completed_units'] for r in rows] == [60, 60, 60, 60, 0]
    assert all(r['error_code'] is None for r in rows)
    assert rows[2]['progress_epoch'] == jobs[2]['progress_epoch']


def test_claim_does_not_recover_live_sibling_and_serializes_overlapping_org_work(store):
    repo, org, actor, gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    other_org = uuid4()
    gateway.execute(query='INSERT INTO identity_organizations(organization_id) VALUES (%(org)s)',
                    parameters=dict(org=other_org))
    gateway.execute(
        query=('INSERT INTO identity_memberships(organization_id,user_id) '
               'VALUES (%(org)s,%(actor)s)'),
        parameters=dict(org=other_org, actor=actor),
    )
    command = WorkRequest('candle_ingestion', 1, ('BTCUSDT',), now-timedelta(hours=2), now)
    first = repo.submit(organization_id=org, actor_user_id=actor, key=uuid4(),
                        request=command, now=now)
    second = repo.submit(organization_id=other_org, actor_user_id=actor, key=uuid4(),
                         request=command, now=now+timedelta(seconds=1))
    with repo.execution() as check:
        assert check is not None
        repo.recover(now=now)
        live = repo.claim(now=now)
        assert live['job_id'] == first['job_id']
        assert repo.claim(now=now+timedelta(minutes=6)) is None
        assert repo.get(organization_id=org, job_id=first['job_id'])['state'] == 'running'
        repo.finish(job=live, state='succeeded', error_code=None, now=now)
        assert repo.claim(now=now)['job_id'] == second['job_id']


@pytest.mark.parametrize('failure', ['submit', 'worker'])
def test_parallel_batch_drains_siblings_before_releasing_lease_on_failure(store, failure):
    from concurrent.futures import Executor
    from dataclasses import dataclass
    from threading import Event
    from types import SimpleNamespace

    from trading.contexts.market_data.application.services.work_requests import (
        MarketDataWorkRequestRunner,
    )

    repo, org, actor, _ = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    jobs = [repo.submit(
        organization_id=org, actor_user_id=actor, key=uuid4(),
        request=WorkRequest('candle_ingestion', 1, (symbol,), now-timedelta(hours=2), now),
        now=now+timedelta(microseconds=i),
    ) for i, symbol in enumerate(('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'))]
    entered, release, failed = Event(), Event(), Event()

    @dataclass
    class Fill:
        writer: Any = None

        def run(self, task):
            entered.set()
            assert release.wait(timeout=10)
            return SimpleNamespace(rows_read=60, rows_written=60)

    runner = MarketDataWorkRequestRunner(repo, cast(Any, Fill()), lambda *_: None,
                                        now=lambda: now, max_windows=1)

    class FaultExecutor(Executor):
        calls = 0

        def submit(self, fn, /, *args, **kwargs):
            self.calls += 1
            if failure == 'submit' and self.calls == 3:
                failed.set()
                raise RuntimeError('injected submit failure')
            if failure == 'worker' and self.calls == 1:
                def fail():
                    failed.set()
                    raise RuntimeError('injected worker failure')
                return workers.submit(fail)
            return workers.submit(fn, *args, **kwargs)

    with ThreadPoolExecutor(4) as workers, ThreadPoolExecutor(1) as coordinator:
        future = coordinator.submit(runner.run_batch, executor=FaultExecutor(), concurrency=4)
        try:
            assert entered.wait(timeout=10) and failed.wait(timeout=10)
            assert not future.done()
            with repo.execution() as rival:
                assert rival is None
        finally:
            release.set()
        with pytest.raises(RuntimeError, match=f'injected {failure} failure'):
            future.result(timeout=10)
    with repo.execution() as successor:
        assert successor is not None
        repo.recover(now=now+timedelta(minutes=6))
    rows = [repo.get(organization_id=org, job_id=j['job_id']) for j in jobs]
    assert all(row['state'] in ('queued', 'retry_wait') for row in rows)
    assert sum(row['completed_units'] for row in rows) == (120 if failure == 'submit' else 180)


def test_parallel_batch_lost_lease_fences_all_four_writers(store):
    from dataclasses import dataclass
    from threading import Barrier, Event
    from types import SimpleNamespace

    from trading.contexts.market_data.application.services.work_requests import (
        MarketDataWorkRequestRunner,
    )

    repo, org, actor, gateway = store
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    jobs = [repo.submit(
        organization_id=org, actor_user_id=actor, key=uuid4(),
        request=WorkRequest('candle_ingestion', 1, (symbol,), now-timedelta(hours=1), now),
        now=now+timedelta(microseconds=i),
    ) for i, symbol in enumerate(('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'))]
    entered, release, writes = Barrier(5), Event(), []

    class Writer:
        def write_1m(self, rows):
            writes.append(rows)

    @dataclass
    class Fill:
        writer: Any

        def run(self, task):
            entered.wait(timeout=10)
            assert release.wait(timeout=10)
            self.writer.write_1m([])
            return SimpleNamespace(rows_read=60, rows_written=60)

    runner = MarketDataWorkRequestRunner(
        repo, cast(Any, Fill(Writer())), lambda *_: None, now=lambda: now,
        retryable_storage_error=lambda error: isinstance(error, psycopg.OperationalError),
    )
    with ThreadPoolExecutor(4) as workers, ThreadPoolExecutor(1) as coordinator:
        future = coordinator.submit(runner.run_batch, executor=workers, concurrency=4)
        try:
            entered.wait(timeout=10)
            key = repo._execution_lock_key
            result = gateway.fetch_one(
                query="""SELECT pg_terminate_backend(pid) terminated FROM pg_locks
                         WHERE locktype='advisory' AND granted
                         AND classid=%(high)s AND objid=%(low)s AND objsubid=1""",
                parameters=dict(high=key >> 32, low=key & 0xffffffff),
            )
            assert result and result['terminated']
        finally:
            release.set()
        assert future.result(timeout=10) == 4
    assert writes == []
    rows = [repo.get(organization_id=org, job_id=j['job_id']) for j in jobs]
    assert all(row['state'] == 'retry_wait' and row['completed_units'] == 0 for row in rows)
