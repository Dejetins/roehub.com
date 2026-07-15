"""Disposable PostgreSQL and ClickHouse proof for research organization isolation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
from datetime import UTC, datetime, timedelta
from statistics import median, pstdev
from typing import Any, Callable, cast
from uuid import UUID, uuid4

import clickhouse_connect
import psycopg

from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobRepository,
    PostgresResearchOrganizationScopeResolver,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.ports.lazy_trades_cache import (
    build_lazy_trades_cache_key,
)
from trading.contexts.backtest.application.services.organization_scoped_market_data import (
    OrganizationScopedCanonicalCandleReader,
)
from trading.contexts.backtest.application.services.research_identity import (
    build_research_content_hash,
    build_research_idempotency_key_hash,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse import (
    ClickHouseCanonicalCandleReader,
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.application.dto import CanonicalCandleBatch1m
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    Symbol,
    TimeRange,
    UserId,
    UtcTimestamp,
)


class ResearchRuntimeProofError(RuntimeError):
    """Raised when disposable research-tenancy evidence is incomplete."""


def _bounded_error_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    message = re.sub(r"https?://[^@\s]+@", "https://[redacted]@", message)
    message = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        "[uuid]",
        message,
    )
    return message[:240] or "unavailable"


def run_probe(*, postgres_dsn: str, clickhouse_dsn: str) -> dict[str, object]:
    now = datetime.now(UTC)
    organization_a, organization_b = uuid4(), uuid4()
    user_a, user_b, ambiguous_user, unscoped_user = (uuid4() for _index in range(4))
    _seed_organizations(
        dsn=postgres_dsn,
        organizations=(organization_a, organization_b),
        users=(user_a, user_b, ambiguous_user, unscoped_user),
        now=now,
    )

    postgres_gateway = PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
    scope_resolver = PostgresResearchOrganizationScopeResolver(
        gateway=postgres_gateway
    )
    repository = PostgresBacktestJobRepository(gateway=postgres_gateway)
    user_id_a = UserId(user_a)
    user_id_b = UserId(user_b)
    organization_id_a = OrganizationId(organization_a)
    organization_id_b = OrganizationId(organization_b)
    scope_a = scope_resolver.resolve(user_id=user_id_a)
    scope_b = scope_resolver.resolve(user_id=user_id_b)
    if (
        scope_a.organization_id != organization_id_a
        or scope_b.organization_id != organization_id_b
    ):
        raise ResearchRuntimeProofError("organization scope was not derived server-side")
    _expect_scope_state(
        resolver=scope_resolver,
        user_id=UserId(ambiguous_user),
        expected_code="research.organization_scope_ambiguous",
    )
    _expect_scope_state(
        resolver=scope_resolver,
        user_id=UserId(unscoped_user),
        expected_code="research.organization_scope_forbidden",
    )

    public_key = "stage09-shared-idempotency-key"
    key_hash_a = build_research_idempotency_key_hash(
        organization_id=organization_id_a,
        idempotency_key=public_key,
    )
    key_hash_b = build_research_idempotency_key_hash(
        organization_id=organization_id_b,
        idempotency_key=public_key,
    )
    if key_hash_a == key_hash_b:
        raise ResearchRuntimeProofError("organization idempotency namespaces collided")

    normalized_request = {
        "coordinates": {"market_id": 1, "symbol": "BTCUSDT"},
        "timeframe": "1m",
        "time_range": {
            "start": "2026-01-01T00:00:00.000Z",
            "end": "2026-01-01T02:00:00.000Z",
        },
        "indicators": [{"id": "ma.sma", "params": {"period": 20}}],
    }
    request_hash_a = build_research_content_hash(payload=normalized_request)
    request_hash_b = build_research_content_hash(
        payload={
            "indicators": [{"params": {"period": 20}, "id": "ma.sma"}],
            "time_range": {
                "end": "2026-01-01T02:00:00.000Z",
                "start": "2026-01-01T00:00:00.000Z",
            },
            "timeframe": "1m",
            "coordinates": {"symbol": "BTCUSDT", "market_id": 1},
        }
    )
    if request_hash_a != request_hash_b:
        raise ResearchRuntimeProofError("organization-neutral content hashes diverged")
    job_a, job_b = uuid4(), uuid4()
    _insert_job(
        dsn=postgres_dsn,
        job_id=job_a,
        organization_id=organization_a,
        user_id=user_a,
        key_hash=key_hash_a,
        request_hash=request_hash_a,
        now=now,
    )
    _insert_job(
        dsn=postgres_dsn,
        job_id=job_b,
        organization_id=organization_b,
        user_id=user_b,
        key_hash=key_hash_b,
        request_hash=request_hash_b,
        now=now,
    )
    stored_job_a = repository.get(
        job_id=job_a,
        organization_id=organization_id_a,
        user_id=user_id_a,
    )
    stored_job_b = repository.get(
        job_id=job_b,
        organization_id=organization_id_b,
        user_id=user_id_b,
    )
    if stored_job_a is None or stored_job_b is None:
        raise ResearchRuntimeProofError("production repository rejected an owned job")
    if repository.get(
        job_id=job_a,
        organization_id=organization_id_b,
        user_id=user_id_b,
    ) is not None:
        raise ResearchRuntimeProofError("cross-organization job query leaked a row")
    if repository.get(
        job_id=job_a,
        organization_id=organization_id_a,
        user_id=user_id_b,
    ) is not None:
        raise ResearchRuntimeProofError("cross-user job query leaked a row")

    postgres_constraints = {
        "job_membership": _expect_postgres_foreign_key(
            dsn=postgres_dsn,
            query="""
                INSERT INTO backtest_jobs (
                    job_id, organization_id, user_id, mode, state, created_at, updated_at,
                    request_json, request_hash, engine_params_hash,
                    backtest_runtime_config_hash, stage, processed_units, total_units, attempt
                ) VALUES (
                    %s, %s, %s, 'template', 'queued', %s, %s,
                    '{"schema":"io.roehub.research-request/v1"}'::jsonb,
                    %s, %s, %s, 'stage_a', 0, 0, 0
                )
            """,
            parameters=(
                uuid4(),
                organization_a,
                user_b,
                now,
                now,
                "b" * 64,
                "c" * 64,
                "d" * 64,
            ),
        ),
        "top_variant_parent": _expect_postgres_foreign_key(
            dsn=postgres_dsn,
            query="""
                INSERT INTO backtest_job_top_variants (
                    organization_id, job_id, rank, variant_key, indicator_variant_key,
                    variant_index, total_return_pct, payload_json, updated_at
                ) VALUES (%s, %s, 1, %s, %s, 0, 0, '{}'::jsonb, %s)
            """,
            parameters=(organization_b, job_a, "b" * 64, "c" * 64, now),
        ),
        "lazy_materialization_parent": _expect_postgres_foreign_key(
            dsn=postgres_dsn,
            query="""
                INSERT INTO backtest_lazy_trades_materializations (
                    task_id, organization_id, owner_user_id, job_id,
                    public_variant_key, variant_hash, request_hash, engine_params_hash,
                    artifact_manifest_hash, cache_key, status, priority_class,
                    created_at, updated_at, attempt, cache_status, ttl_seconds
                ) VALUES (
                    %s, %s, %s, %s, 'stage09', %s, %s, %s, %s, %s,
                    'queued', 'interactive', %s, %s, 0, 'miss', 60
                )
            """,
            parameters=(
                uuid4(),
                organization_b,
                user_b,
                job_a,
                "e" * 64,
                request_hash_a,
                "f" * 64,
                "1" * 64,
                "2" * 64,
                now,
                now,
            ),
        ),
    }

    client = clickhouse_connect.get_client(
        dsn=clickhouse_dsn,
        database="default",
        connect_timeout=10,
    )
    try:
        clickhouse_database = _resolve_clickhouse_database(client)
        canonical_table = f"{clickhouse_database}.canonical_candles_1m"
        candle_open = now.replace(second=0, microsecond=0)
        candle_count = 120
        candle_rows: list[list[object]] = []
        for index in range(candle_count):
            row_open = candle_open + timedelta(minutes=index)
            row_close = row_open + timedelta(minutes=1) - timedelta(milliseconds=1)
            base_price = 100.0 + float(index) / 10.0
            candle_rows.append(
                [
                    1,
                    "BTCUSDT",
                    "binance:spot:BTCUSDT",
                    row_open,
                    row_close,
                    base_price,
                    base_price + 2.0,
                    base_price - 1.0,
                    base_price + 1.0,
                    10.0 + float(index),
                    (10.0 + float(index)) * (base_price + 1.0),
                    5 + index,
                    6.0,
                    606.0,
                    "rest",
                    now,
                    uuid4(),
                ]
            )
        client.insert(
            canonical_table,
            candle_rows,
            column_names=[
                "market_id",
                "symbol",
                "instrument_key",
                "ts_open",
                "ts_close",
                "open",
                "high",
                "low",
                "close",
                "volume_base",
                "volume_quote",
                "trades_count",
                "taker_buy_volume_base",
                "taker_buy_volume_quote",
                "source",
                "ingested_at",
                "ingest_id",
            ],
        )
        direct_reader = ClickHouseCanonicalCandleReader(
            gateway=ClickHouseConnectGateway(client),
            database=clickhouse_database,
        )
        scoped_reader = OrganizationScopedCanonicalCandleReader(
            scope_resolver=scope_resolver,
            canonical_reader=direct_reader,
        )
        instrument_id = InstrumentId(
            market_id=MarketId(1),
            symbol=Symbol("BTCUSDT"),
        )
        time_range = TimeRange(
            start=UtcTimestamp(candle_open),
            end=UtcTimestamp(candle_open + timedelta(minutes=candle_count)),
        )

        def read_direct() -> CanonicalCandleBatch1m:
            return direct_reader.read_1m_arrays(instrument_id, time_range)

        def read_scoped() -> CanonicalCandleBatch1m:
            return scoped_reader.read_1m_arrays(
                user_id=user_id_a,
                instrument_id=instrument_id,
                time_range=time_range,
            ).candles

        direct_batch = read_direct()
        scoped_batch_a = read_scoped()
        scoped_result_b = scoped_reader.read_1m_arrays(
            user_id=user_id_b,
            instrument_id=instrument_id,
            time_range=time_range,
        )
        if direct_batch.row_count() != candle_count:
            raise ResearchRuntimeProofError("shared canonical candle count is incomplete")
        direct_digest = _candle_batch_digest(direct_batch)
        if (
            direct_digest != _candle_batch_digest(scoped_batch_a)
            or direct_digest != _candle_batch_digest(scoped_result_b.candles)
        ):
            raise ResearchRuntimeProofError("shared candle parity failed")
        if scoped_result_b.scope.organization_id != organization_id_b:
            raise ResearchRuntimeProofError("shared candle scope was not server-derived")
        _expect_scoped_candle_state(
            reader=scoped_reader,
            user_id=UserId(ambiguous_user),
            instrument_id=instrument_id,
            time_range=time_range,
            expected_code="research.organization_scope_ambiguous",
        )
        _expect_scoped_candle_state(
            reader=scoped_reader,
            user_id=UserId(unscoped_user),
            instrument_id=instrument_id,
            time_range=time_range,
            expected_code="research.organization_scope_forbidden",
        )
        stored_rows = int(
            client.query(
                """
                SELECT count()
                FROM {canonical_table} FINAL
                WHERE market_id = 1
                  AND symbol = 'BTCUSDT'
                  AND ts_open >= %(start)s
                  AND ts_open < %(end)s
                """.format(canonical_table=canonical_table),
                parameters={
                    "start": candle_open,
                    "end": candle_open + timedelta(minutes=candle_count),
                },
            ).result_rows[0][0]
        )
        if stored_rows != candle_count:
            raise ResearchRuntimeProofError("canonical candles were duplicated per organization")

        direct_ms, scoped_ms = _measure_comparable_ms(
            baseline=read_direct,
            candidate=read_scoped,
            warmups=10,
            repeats=50,
        )
        p95_overhead_ms = round(scoped_ms["p95"] - direct_ms["p95"], 3)
        p95_overhead_budget_ms = 15.0
        if p95_overhead_ms > p95_overhead_budget_ms:
            raise ResearchRuntimeProofError("organization authorization p95 overhead exceeded")
    finally:
        client.close()

    cache_key_values = {
        "job_id": str(job_a),
        "variant_key": "stage09",
        "variant_hash": "d" * 64,
        "request_hash": request_hash_a,
        "engine_params_hash": "e" * 64,
        "artifact_manifest_hash": "f" * 64,
    }
    cache_hash_a = build_lazy_trades_cache_key(
        organization_id=str(organization_id_a),
        **cache_key_values,
    ).digest
    cache_hash_b = build_lazy_trades_cache_key(
        organization_id=str(organization_id_b),
        **cache_key_values,
    ).digest
    if cache_hash_a == cache_hash_b:
        raise ResearchRuntimeProofError("organization cache namespaces collided")

    return {
        "schema": "io.roehub.research-tenancy-runtime-proof/v1alpha1",
        "server_derived_scope": "passed",
        "ambiguous_scope": "rejected",
        "missing_scope": "rejected",
        "cross_organization_repository_read": "rejected",
        "database_constraints": postgres_constraints,
        "organization_idempotency_namespace": "passed",
        "organization_cache_namespace": "passed",
        "request_hash_parity": "passed",
        "request_hash_pipeline": "build_research_content_hash",
        "production_repository_adapter": "PostgresBacktestJobRepository",
        "production_scope_resolver": "PostgresResearchOrganizationScopeResolver",
        "production_candle_reader": "OrganizationScopedCanonicalCandleReader",
        "shared_canonical_rows": candle_count,
        "shared_candle_parity": "passed",
        "direct_read_ms": direct_ms,
        "scoped_read_ms": scoped_ms,
        "authorization_p95_overhead_ms": p95_overhead_ms,
        "authorization_p95_budget_ms": p95_overhead_budget_ms,
        "benchmark_method": "alternating warm-cache reads; 10 warmups; 50 samples per path",
        "authorization_overhead": "passed",
    }


def _resolve_clickhouse_database(client: Any) -> str:
    rows = client.query(
        """
        SELECT database
        FROM system.tables
        WHERE name = 'canonical_candles_1m'
        ORDER BY database
        """
    ).result_rows
    if len(rows) != 1:
        raise ResearchRuntimeProofError("canonical ClickHouse source is not unique")
    database = str(rows[0][0])
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,62}", database) is None:
        raise ResearchRuntimeProofError("canonical ClickHouse database is invalid")
    return database


def _seed_organizations(
    *,
    dsn: str,
    organizations: tuple[UUID, UUID],
    users: tuple[UUID, UUID, UUID, UUID],
    now: datetime,
) -> None:
    organization_a, organization_b = organizations
    user_a, user_b, ambiguous_user, unscoped_user = users
    with psycopg.connect(dsn, autocommit=False) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT installation_id FROM identity_installations WHERE singleton_key = TRUE"
        )
        row = cursor.fetchone()
        if row is None:
            raise ResearchRuntimeProofError("disposable installation is unavailable")
        installation_id = cast(UUID, row[0])
        cursor.executemany(
            """
            INSERT INTO identity_users (
                user_id, telegram_user_id, paid_level, created_at,
                last_login_at, is_deleted, keycloak_subject
            ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
            """,
            [(user_id, now, now) for user_id in users],
        )
        cursor.executemany(
            """
            INSERT INTO identity_organizations (
                organization_id, installation_id, slug, display_name, status, created_at
            ) VALUES (%s, %s, %s, %s, 'active', %s)
            """,
            (
                (
                    organization_a,
                    installation_id,
                    f"stage09-a-{organization_a.hex[:8]}",
                    "Stage 09 A",
                    now,
                ),
                (
                    organization_b,
                    installation_id,
                    f"stage09-b-{organization_b.hex[:8]}",
                    "Stage 09 B",
                    now,
                ),
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_memberships (
                organization_id, user_id, role, status, created_at, updated_at
            ) VALUES (%s, %s, %s, 'active', %s, %s)
            """,
            (
                (organization_a, user_a, "owner", now, now),
                (organization_b, user_b, "owner", now, now),
                (organization_a, ambiguous_user, "viewer", now, now),
                (organization_b, ambiguous_user, "viewer", now, now),
            ),
        )
        _ = unscoped_user


def _expect_scope_state(
    *,
    resolver: PostgresResearchOrganizationScopeResolver,
    user_id: UserId,
    expected_code: str,
) -> None:
    try:
        resolver.resolve(user_id=user_id)
    except RoehubError as error:
        if error.code == expected_code:
            return
        raise
    raise ResearchRuntimeProofError("research scope unexpectedly resolved")


def _expect_scoped_candle_state(
    *,
    reader: OrganizationScopedCanonicalCandleReader,
    user_id: UserId,
    instrument_id: InstrumentId,
    time_range: TimeRange,
    expected_code: str,
) -> None:
    try:
        reader.read_1m_arrays(
            user_id=user_id,
            instrument_id=instrument_id,
            time_range=time_range,
        )
    except RoehubError as error:
        if error.code == expected_code:
            return
        raise
    raise ResearchRuntimeProofError("shared candle access unexpectedly resolved")


def _insert_job(
    *,
    dsn: str,
    job_id: UUID,
    organization_id: UUID,
    user_id: UUID,
    key_hash: str,
    request_hash: str,
    now: datetime,
) -> None:
    request = json.dumps(
        {
            "schema": "io.roehub.research-request/v1",
            "idempotency": {"key_hash": key_hash},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with psycopg.connect(dsn, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO backtest_jobs (
                job_id, organization_id, user_id, mode, state, created_at, updated_at,
                request_json, request_hash, engine_params_hash,
                backtest_runtime_config_hash,
                stage, processed_units, total_units, attempt
            ) VALUES (
                %s, %s, %s, 'template', 'queued', %s, %s,
                %s::jsonb, %s, %s, %s, 'stage_a', 0, 0, 0
            )
            """,
            (
                job_id,
                organization_id,
                user_id,
                now,
                now,
                request,
                request_hash,
                "b" * 64,
                "c" * 64,
            ),
        )


def _expect_postgres_foreign_key(
    *,
    dsn: str,
    query: str,
    parameters: tuple[object, ...],
) -> str:
    try:
        with psycopg.connect(dsn, autocommit=False) as connection, connection.cursor() as cursor:
            cursor.execute(cast(Any, query), parameters)
    except psycopg.Error as error:
        if error.sqlstate == "23503":
            return "passed"
        raise ResearchRuntimeProofError("unexpected PostgreSQL constraint error") from error
    raise ResearchRuntimeProofError("cross-organization PostgreSQL write was accepted")


def _candle_batch_digest(batch: CanonicalCandleBatch1m) -> str:
    digest = hashlib.sha256()
    for array in (batch.open_time_ms, batch.close_time_ms, batch.ohlcv_f32):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _measure_comparable_ms(
    *,
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _index in range(warmups):
        baseline()
        candidate()
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(repeats):
        ordered_callbacks = (
            ((baseline, baseline_samples), (candidate, candidate_samples))
            if index % 2 == 0
            else ((candidate, candidate_samples), (baseline, baseline_samples))
        )
        for callback, samples in ordered_callbacks:
            started = time.perf_counter()
            callback()
            samples.append((time.perf_counter() - started) * 1000.0)
    return (
        _summarize_ms(baseline_samples, warmups=warmups),
        _summarize_ms(candidate_samples, warmups=warmups),
    )


def _summarize_ms(samples: list[float], *, warmups: int) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]
    return {
        "min": round(ordered[0], 3),
        "p50": round(median(ordered), 3),
        "p95": round(p95, 3),
        "max": round(ordered[-1], 3),
        "stddev": round(pstdev(ordered), 3),
        "warmups": float(warmups),
        "samples": float(len(ordered)),
    }


def main() -> int:
    if os.environ.get("ROEHUB_DISPOSABLE_STORAGE_PROOF") != "1":
        print("research tenancy runtime proof failed: disposable proof flag is required")
        return 1
    postgres_dsn = os.environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "").strip()
    clickhouse_dsn = os.environ.get("ROEHUB_STORAGE_CLICKHOUSE_DSN", "").strip()
    if not postgres_dsn or not clickhouse_dsn:
        print("research tenancy runtime proof failed: storage endpoints are unavailable")
        return 1
    try:
        result = run_probe(
            postgres_dsn=postgres_dsn,
            clickhouse_dsn=clickhouse_dsn,
        )
    except Exception as error:  # noqa: BLE001
        frames = traceback.extract_tb(error.__traceback__)
        local_frames = [
            frame for frame in frames if frame.filename.endswith("research_runtime_probe.py")
        ]
        location = local_frames[-1] if local_frames else (frames[-1] if frames else None)
        bounded_location = (
            f"{location.name}:{location.lineno}" if location is not None else "unknown"
        )
        print(
            "research tenancy runtime proof failed: "
            f"{type(error).__name__} code={getattr(error, 'code', 'unknown')} "
            f"at {bounded_location}; detail={_bounded_error_message(error)}"
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
