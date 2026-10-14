"""Counted PostgreSQL operational observer; no unbounded query text or secret labels."""

from __future__ import annotations

import os
import threading
from typing import Any

from trading.platform.postgres_budget import connect, reap, reservation_counts

_lock = threading.Lock()
_connection: Any = None
_metrics = ""
_maintenance: Any = None
_maintenance_cursor: str | None = None


def sample() -> bool:
    global _connection, _metrics, _maintenance, _maintenance_cursor
    with _lock:
        budget_metrics = ""
        try:
            budget_metrics = "".join(
                f'roehub_postgres_reservations_{name}{{class="{kind}"}} {value}\n'
                for kind, counts in reservation_counts().items()
                for name, value in counts.items()
            )
            if _connection is None or _connection.closed:
                _connection = connect(
                    os.environ["ROEHUB_PG_MONITOR_DSN"],
                    budget_class="admin",
                    autocommit=True,
                    options="-c statement_timeout=2000 -c lock_timeout=250",
                )
            reap(_connection)
            from apps.migrations.market_data import provision_requested_partitions

            schema_ready = _connection.execute(
                "SELECT to_regclass('market_data.canonical_candles_1m') IS NOT NULL"
            ).fetchone()[0]
            if schema_ready:
                if _maintenance is None or _maintenance.closed:
                    _maintenance = connect(
                        os.environ["ROEHUB_PG_MAINTENANCE_DSN"], budget_class="admin"
                    )
                try:
                    with _maintenance.transaction():
                        _, next_cursor = provision_requested_partitions(
                            _maintenance, after_job_id=_maintenance_cursor
                        )
                    _maintenance_cursor = next_cursor
                except Exception:
                    # A missing grant or interrupted DDL is degraded, never monitor churn.
                    schema_ready = False
            rows = _connection.execute(
                "SELECT CASE WHEN application_name LIKE 'rh-market-%%' THEN 'market' "
                "WHEN application_name LIKE 'rh-control-%%' THEN 'control' ELSE 'admin' END, "
                "count(*), count(*) FILTER(WHERE wait_event_type='Lock'), "
                "count(*) FILTER(WHERE xact_start < now()-interval '30 seconds') "
                "FROM pg_stat_activity WHERE backend_type='client backend' GROUP BY 1"
            ).fetchall()
            lines = []
            for kind, count, locks, long_transactions in rows:
                for name, value in (
                    ("connections", count),
                    ("lock_waiters", locks),
                    ("long_transactions", long_transactions),
                ):
                    lines.append(f'roehub_postgres_{name}{{class="{kind}"}} {value}')
            wal = _connection.execute(
                "SELECT wal_bytes,wal_buffers_full FROM pg_stat_wal"
            ).fetchone()
            lines.extend(
                (
                    f"roehub_postgres_wal_bytes_total {wal[0]}",
                    f"roehub_postgres_wal_buffers_full_total {wal[1]}",
                )
            )
            checkpoint = _connection.execute(
                "SELECT checkpoints_timed, checkpoints_req FROM pg_stat_bgwriter"
            ).fetchone()
            lines.extend(
                (
                    f"roehub_postgres_checkpoints_timed_total {checkpoint[0]}",
                    f"roehub_postgres_checkpoints_requested_total {checkpoint[1]}",
                )
            )
            dead = _connection.execute(
                "SELECT coalesce(sum(n_dead_tup),0), coalesce(sum(autovacuum_count),0) "
                "FROM pg_stat_user_tables"
            ).fetchone()
            lines.extend(
                (
                    f"roehub_postgres_dead_tuples {dead[0]}",
                    f"roehub_postgres_autovacuums_total {dead[1]}",
                )
            )
            _metrics = budget_metrics + "\n".join(lines) + "\nroehub_postgres_observer_up 1\n"
            return bool(schema_ready)
        except Exception:
            if _connection is not None:
                _connection.close()
            _connection = None
            _metrics = budget_metrics + "roehub_postgres_observer_up 0\n"
            return False


def metrics_text() -> str:
    with _lock:
        return _metrics
