"""Lease/socket status integration proof in a disposable PostgreSQL database."""
import os
import unittest
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
    PostgresStreamRecoveryStore,
)


@unittest.skipUnless(os.environ.get("IDENTITY_PG_DSN"), "PostgreSQL DSN required")
class StreamConnectionsPostgresTests(unittest.TestCase):
    def test_socket_status_is_scoped_fresh_and_fenced(self):
        base = os.environ["IDENTITY_PG_DSN"]
        database = "stream_connections_test_" + uuid4().hex
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            try:
                dsn = make_conninfo(base, dbname=database)
                root = Path(__file__).resolve().parents[2]
                with psycopg.connect(dsn) as conn:
                    for name in ("0027_market_data_stream_recovery_v1.sql",
                                 "0033_market_data_stream_connections_v1.sql",
                                 "0033_market_data_stream_connections_v1.sql"):
                        conn.execute((root / "migrations/postgres" / name).read_bytes())
                store = PostgresStreamRecoveryStore(dsn, max_entries=2)
                with self.assertRaisesRegex(RuntimeError, "lease required"):
                    store.publish_connections({(1, "ETHUSDT"): True})
                with store.lease():
                    store.publish_connections({(1, "ETHUSDT"): False, (3, "ETHUSDT"): True})
                    self.assertEqual(store.collector_status()["connections"],
                                     {(1, "ETHUSDT"): False, (3, "ETHUSDT"): True})
                    with psycopg.connect(dsn) as conn:
                        conn.execute("UPDATE market_data_stream_connections "
                                     "SET observed_at=now()-interval '21 seconds'")
                    self.assertEqual(store.collector_status()["connections"], {})
                    store.publish_connections({(1, "ETHUSDT"): True})
                    self.assertEqual(store.collector_status()["connections"],
                                     {(1, "ETHUSDT"): True})
                    with self.assertRaisesRegex(RuntimeError, "capacity"):
                        store.publish_connections({(1, "A"): True, (1, "B"): True, (1, "C"): True})
                self.assertEqual(store.collector_status()["connections"], {})
                with store.lease():
                    self.assertEqual(store.collector_status()["connections"], {})
            finally:
                admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


if __name__ == "__main__":
    unittest.main()
