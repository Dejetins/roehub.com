"""Historical stream demand reaches maintenance without granting application DDL."""

from datetime import datetime, timezone
from uuid import uuid4

import psycopg

from apps.migrations.market_data import provision_requested_partitions
from tests.real_infra.market_data.test_storage_schema_postgres import db  # noqa: F401
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_errors import (
    transient_postgres_error,
)


def test_stream_repair_provisions_missing_month_and_classifies_only_partition_failure(db):  # noqa: F811
    with psycopg.connect(db) as conn:
        try:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO market_data.raw_binance_klines_1m "
                    "(market_id,symbol,open_time) VALUES (1,'TEST', '1982-03-01')"
                )
        except psycopg.errors.CheckViolation as error:
            assert transient_postgres_error(error) == "storage_partition_pending"
        else:
            raise AssertionError("missing month was accepted")
        conn.execute(
            "INSERT INTO market_data_stream_recovery "
            "(task_key,token,market_id,symbol,start_at,end_at,reason,state,error_code) "
            "VALUES (%s,%s,1,'TEST',%s,%s,'stream_history_repair','failed',"
            "'non_retryable_ingestion_error')",
            (
                "partition-proof",
                uuid4(),
                datetime(1982, 3, 1, tzinfo=timezone.utc),
                datetime(1982, 4, 1, tzinfo=timezone.utc),
            ),
        )
        count, cursor = provision_requested_partitions(conn)
        assert count == 1 and cursor is not None
        partition = conn.execute(
            "SELECT to_regclass('market_data.raw_binance_klines_1m_198203')"
        ).fetchone()
        assert partition is not None and partition[0]
        assert provision_requested_partitions(conn)[0] == 0
        try:
            with conn.transaction():
                conn.execute(
                    "INSERT INTO market_data_stream_recovery "
                    "(task_key,token,market_id,symbol,start_at,end_at,reason) "
                    "VALUES ('bad',%s,1,'TEST',now(),now()-interval '1 hour','test')",
                    (uuid4(),),
                )
        except psycopg.errors.CheckViolation as error:
            assert transient_postgres_error(error) is None
        conn.rollback()


def test_contending_history_verification_is_retryable(db):  # noqa: F811
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
        PostgresHistoryRangeVerifier,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (  # noqa: E501
        IngestionGateway,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
        CoverageUnavailable,
    )
    from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

    gateway = IngestionGateway(db)
    verifier = PostgresHistoryRangeVerifier(gateway)
    instrument = InstrumentId(MarketId(1), Symbol("LEASETEST"))
    first = verifier.acquire(instrument_id=instrument)
    try:
        try:
            verifier.acquire(instrument_id=instrument)
        except CoverageUnavailable as error:
            assert transient_postgres_error(error) == "storage_busy"
        else:
            raise AssertionError("concurrent verification lease granted")
        assert transient_postgres_error(CoverageUnavailable("stale_source_generation")) is None
    finally:
        verifier.release(instrument_id=instrument, fence=first)
        gateway.close()
