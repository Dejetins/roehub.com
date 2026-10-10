"""Run with IDENTITY_PG_DSN against PostgreSQL. Uses and removes a disposable schema."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from trading.contexts.identity.adapters.outbound.persistence.postgres.gateway import (
    PsycopgIdentityPostgresGateway,
)
from trading.contexts.identity.adapters.outbound.persistence.postgres.session_repository import (
    PostgresIdentitySessionRepository,
)
from trading.shared_kernel.primitives import UserId


@unittest.skipUnless(os.environ.get("IDENTITY_PG_DSN"), "PostgreSQL DSN required")
class BrowserDevicePostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dsn = os.environ["IDENTITY_PG_DSN"]
        cls.schema = "device_qa_" + uuid4().hex
        with psycopg.connect(cls.base_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.dsn = make_conninfo(cls.base_dsn, options=f"-c search_path={cls.schema}")
        root = Path(__file__).resolve().parents[2]
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(
                "CREATE TABLE identity_users(user_id UUID PRIMARY KEY, telegram_user_id bigint)"
            )
            for filename in (
                "0005_identity_keycloak_cutover_v1.sql",
                "0032_identity_browser_devices_v1.sql",
            ):
                connection.execute(
                    (root / "migrations/postgres" / filename).read_bytes(), prepare=False
                )
            # Re-application is non-destructive.
            connection.execute(
                (root / "migrations/postgres/0032_identity_browser_devices_v1.sql").read_bytes(),
                prepare=False,
            )

    @classmethod
    def tearDownClass(cls):
        with psycopg.connect(cls.base_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        self.repo = PostgresIdentitySessionRepository(
            gateway=PsycopgIdentityPostgresGateway(dsn=self.dsn)
        )
        self.owner = UserId(uuid4())
        self.now = datetime.now(timezone.utc)
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "INSERT INTO identity_users(user_id) VALUES (%s)", (str(self.owner),)
            )

    def stored(self, session_id):
        session = self.repo.find_by_session_id(session_id=session_id)
        assert session is not None
        return session

    def session(self):
        return self.repo.create_session(
            user_id=self.owner, now=self.now, idle_ttl_seconds=3600, absolute_ttl_seconds=7200
        )

    def test_concurrent_logins_leave_one_device_and_one_live_credential(self):
        sessions = [self.session() for _ in range(8)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(
                executor.map(
                    lambda s: self.repo.bind_device(
                        session_id=s.session_id,
                        device_key="a" * 64,
                        label="Chrome · macOS",
                        now=self.now,
                    ),
                    sessions,
                )
            )
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(
            sum(
                self.stored(s.session_id).is_active_at(at=self.now)
                for s in sessions
            ),
            1,
        )
        rows = self.repo.list_devices(user_id=self.owner, now=self.now, before=None, limit=20)
        self.assertEqual(len(rows), 1)
        self.repo.revoke_device(user_id=self.owner, device_id=ids[0], now=self.now)
        self.assertFalse(
            self.repo.list_devices(user_id=self.owner, now=self.now, before=None, limit=20)[
                0
            ].active
        )
        new = self.session()
        self.assertEqual(
            self.repo.bind_device(
                session_id=new.session_id, device_key="a" * 64, label="Chrome", now=self.now
            ),
            ids[0],
        )

    def test_activity_and_owner_isolation(self):
        session = self.session()
        did = self.repo.bind_device(
            session_id=session.session_id, device_key="b" * 64, label="Firefox", now=self.now
        )
        later = self.now + timedelta(minutes=2)
        self.repo.touch_session(session_id=session.session_id, now=later)
        touched = self.repo.find_by_session_id(session_id=session.session_id)
        assert touched is not None
        self.assertEqual(touched.last_seen_at, later)
        self.assertEqual(touched.idle_expires_at, session.idle_expires_at)
        self.assertEqual(touched.created_at, session.created_at)
        self.repo.revoke_device(user_id=UserId(uuid4()), device_id=did, now=later)
        self.assertTrue(
            self.stored(session.session_id).is_active_at(at=later)
        )
        self.repo.revoke_device(user_id=self.owner, device_id=did, now=later)
        self.repo.touch_session(session_id=session.session_id, now=later + timedelta(minutes=2))
        self.assertEqual(
            self.stored(session.session_id).last_seen_at, later
        )
        self.assertFalse(
            self.stored(session.session_id).is_active_at(at=later)
        )

    def test_legacy_revoke_and_stable_pagination(self):
        legacy = self.session()
        devices = [
            self.repo.bind_device(
                session_id=self.session().session_id,
                device_key=uuid4().hex * 2,
                label="Browser",
                now=self.now,
            )
            for _ in range(4)
        ]
        first = self.repo.list_devices(user_id=self.owner, now=self.now, before=None, limit=2)
        second = self.repo.list_devices(
            user_id=self.owner, now=self.now, before=first[-1].device_id, limit=2
        )
        self.assertEqual({r.device_id for r in first + second}, set(devices))
        self.assertEqual(self.repo.legacy_session_count(user_id=self.owner, now=self.now), 1)
        self.repo.revoke_device(user_id=self.owner, device_id=None, now=self.now)
        self.assertFalse(
            self.stored(legacy.session_id).is_active_at(at=self.now)
        )
        self.assertTrue(
            all(
                r.active
                for r in self.repo.list_devices(
                    user_id=self.owner, now=self.now, before=None, limit=20
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
