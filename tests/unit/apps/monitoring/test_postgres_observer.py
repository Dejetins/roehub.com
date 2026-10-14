"""Operational metrics keep reservations visible and sanitize failed queries."""

from types import SimpleNamespace

from apps.monitoring import postgres_observer as observer


def test_failed_sample_keeps_budget_without_driver_message(monkeypatch):
    monkeypatch.setattr(observer, "_connection", None)
    monkeypatch.setattr(observer, "reservation_counts", lambda: {"market": {"active": 2}})
    monkeypatch.setenv("ROEHUB_PG_MONITOR_DSN", "private")

    def fail(*args, **kwargs):
        assert kwargs["budget_class"] == "admin"
        raise ConnectionError("private credential")

    monkeypatch.setattr(observer, "connect", fail)
    assert observer.sample() is False
    text = observer.metrics_text()
    assert 'roehub_postgres_reservations_active{class="market"} 2' in text
    assert "roehub_postgres_observer_up 0" in text
    assert "private" not in text


def test_successful_sample_exports_pg_families(monkeypatch):
    from apps.migrations import market_data

    class Connection:
        closed = False

        def execute(self, query):
            if "to_regclass" in query:
                return SimpleNamespace(fetchone=lambda: (True,))
            if "pg_stat_activity" in query:
                return SimpleNamespace(fetchall=lambda: [("market", 3, 1, 2)])
            return SimpleNamespace(fetchone=lambda: (5, 7))

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(observer, "_connection", Connection())
    monkeypatch.setattr(
        observer, "_maintenance", SimpleNamespace(closed=False, transaction=Transaction)
    )
    monkeypatch.setattr(observer, "reservation_counts", lambda: {})
    monkeypatch.setattr(observer, "reap", lambda connection: None)
    monkeypatch.setattr(market_data, "provision_requested_partitions", lambda *a, **k: (0, None))
    assert observer.sample()
    text = observer.metrics_text()
    for family in (
        "connections",
        "lock_waiters",
        "long_transactions",
        "wal_bytes_total",
        "checkpoints_requested_total",
        "dead_tuples",
        "autovacuums_total",
    ):
        assert "roehub_postgres_" + family in text
    assert "roehub_postgres_observer_up 1" in text
