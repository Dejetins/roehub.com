"""Transactional lifecycle events, tested in a disposable schema only."""

import os
import unittest
from pathlib import Path
from typing import LiteralString, cast
from uuid import uuid4

import psycopg
from psycopg import sql


@unittest.skipUnless(
    os.environ.get("ROEHUB_MARKET_DATA_TEST_DSN"), "explicit PostgreSQL target required"
)
class StreamEventsTest(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ["ROEHUB_MARKET_DATA_TEST_DSN"], autocommit=True)
        self.schema = "stream_events_test_" + uuid4().hex
        self.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
        for name in [
            "0027_market_data_stream_recovery_v1.sql",
            "0033_market_data_stream_connections_v1.sql",
            "0035_market_data_stream_events_v1.sql",
            "0035_market_data_stream_events_v1.sql",
        ]:
            self.conn.execute(
                sql.SQL(cast(LiteralString, Path("migrations/postgres", name).read_text()))
            )

    def tearDown(self):
        self.conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema)))
        self.conn.close()

    def events(self):
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT event_type FROM market_data_stream_events ORDER BY event_id"
            ).fetchall()
        ]

    def test_transitions_not_heartbeats_and_rollback(self):
        self.conn.execute(
            "INSERT INTO market_data_stream_connections VALUES (1,'BTCUSDT',true,1,now())"
        )
        self.conn.execute("UPDATE market_data_stream_connections SET observed_at=now()")
        self.conn.execute("UPDATE market_data_stream_connections SET connected=false")
        self.conn.execute("UPDATE market_data_stream_connections SET connected=true")
        self.assertEqual(
            self.events(), ["connection_opened", "connection_lost", "connection_opened"]
        )
        try:
            with self.conn.transaction():
                self.conn.execute("UPDATE market_data_stream_connections SET connected=false")
                raise RuntimeError("rollback")
        except RuntimeError:
            pass
        self.assertEqual(len(self.events()), 3)

    def test_recovery_events_are_sparse_and_exclude_candle_receipts(self):
        for reason in ["ws_recovery", "reconnect_tail"]:
            self.conn.execute(
                """INSERT INTO market_data_stream_recovery
                (task_key,token,market_id,symbol,start_at,end_at,reason)
                VALUES (%s,%s,1,'BTCUSDT',now(),now()+interval '1 hour',%s)""",
                (reason, uuid4(), reason),
            )
        self.assertEqual(self.events(), ["recovery_queued"])
        for attempt in range(1, 9):
            self.conn.execute(
                "UPDATE market_data_stream_recovery SET attempt=%s,error_code='source_unavailable'",
                (attempt,),
            )
        self.assertEqual(self.events().count("recovery_waiting"), 4)
        self.conn.execute(
            "UPDATE market_data_stream_recovery SET start_at=start_at+interval '5 minutes',"
            "attempt=0,error_code=NULL"
        )
        self.assertEqual(self.events()[-1], "recovery_progress")
        self.conn.execute("DELETE FROM market_data_stream_recovery")
        self.assertEqual(self.events()[-1], "recovery_closed")

    def test_retention_prunes_expired_events_at_amortized_boundary(self):
        self.conn.execute(
            "INSERT INTO market_data_stream_events"
            " (market_id,symbol,event_type,occurred_at)"
            " VALUES (1,'BTCUSDT','connection_opened',now()-interval '31 days')"
        )
        self.conn.execute(
            "SELECT setval(pg_get_serial_sequence('market_data_stream_events','event_id'),1023)"
        )
        self.conn.execute(
            "INSERT INTO market_data_stream_events (market_id,symbol,event_type)"
            " VALUES (1,'BTCUSDT','connection_opened')"
        )
        self.assertEqual(
            self.conn.execute("SELECT event_id FROM market_data_stream_events").fetchall(),
            [(1024,)],
        )


if __name__ == "__main__":
    unittest.main()
